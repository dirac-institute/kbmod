import warnings
import numpy as np

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import kbmod.trajectory_generator
import kbmod.wcs_utils
import kbmod.reprojection_utils

from kbmod.filters.known_object_filters import KnownObjsMatcher
from kbmod.results import Results


try:
    from lsst.daf.butler import DatasetId
    from lsst.source.injection import VisitInjectConfig, VisitInjectTask

    HAS_LSST = True
except ImportError:
    HAS_LSST = False


def generate_injection_catalog(
    ic,
    search_config,
    global_wcs,
    n_objs_per_ic=50,
    guess_distance=None,
    mag_range=(19.0, 26.0),
    source_type="Star",
):
    """Generate an injection catalog for an ImageCollection to be consumed by `inject_sources_into_ic`.

    Incorporates sub-pixel and sub-search-velocity-resolution jitter, and handles coordinate propagation
    keeping objects moving in straight lines inside global WCS pixel coordinates.

    If no heliocentric guess distance is provided, the catalog will provide the coordinates in the original
    reference frame defined by `global_wcs`, which is presumed to not be reflex-corrected.

    If a guess distance is provided, the catalog will provide the coordinates in EBD which is presumed to be
    defined by `global_wcs` reflex-corrected at the specified guess distance. The injection coordinates will be
    inverse parallax-corrected and should no longer necessarily appear as straight "lines" until after
    the injection and resampling process back to EBD at the same guess distance.

    Parameters
    ----------
    ic : `kbmod.ImageCollection`
       Image collection to inject into.
    search_config : `kbmod.configuration.SearchConfiguration`
       Search config object.
    global_wcs : `astropy.wcs.WCS`
       Shared astropy WCS object. If guess_distance is provided,
       this should already be in the reflex-corrected reference frame.
    n_objs_per_ic : `int`, default: 50
       Number of objects to simulate.
    guess_distance : `float` or None, default: None
       Guess distance (AU) used for inverse parallax correction.
    mag_range : `tuple`, default: (19.0, 26.0)
       Magnitude sampling bounds (min, max) in
    source_type : `str`, default: "Star"
       Source type designation in the injection catalog.

    Returns
    -------
    catalog : astropy.table.Table
        Coordinates and magnitudes of simulated objects. Its columns are defined as:
        - injection_id: unique identifier for each injection
        - ra: right ascension of the object in degrees for injection
        - dec: declination of the object in degrees for injection
        - mag: magnitude of the object
        - guess_distance: guess distance (AU) used for inverse parallax correction if provided
        - source_type: source type designation in the injection catalog, e.g. "Star"
        - obj_ids: unique identifier for each injected object
        - obstime: observation time in MJD, aligning with ic["mjd_mid"]
        - plot_x: x-coordinate of the object in the global WCS frame for plotting convenience
        - plot_y: y-coordinate of the object in the global WCS frame for plotting convenience
        - ra_{guess_distance}: right ascension of the object in the global WCS frame at the guess distance
        - dec_{guess_distance}: declination of the object in the global WCS frame at the guess distance
    """
    # Define trajectories along the angle of the ecliptic for each synthetic object to inject
    eclip_angle = kbmod.wcs_utils.calc_ecliptic_angle(global_wcs)
    trjgen = kbmod.trajectory_generator.create_trajectory_generator(
        search_config["generator_config"], given_ecliptic=eclip_angle
    )
    candidates = [trj for trj in trjgen]
    trjs = np.random.choice(candidates, n_objs_per_ic)

    # Determine the boundaries of the global WCS frame. Note that this is presumed to be a common
    # reference frame for all exposures in the image collection.
    try:
        pixel_boundaries = global_wcs.world_to_pixel(
            SkyCoord(global_wcs.calc_footprint()[:, 0], global_wcs.calc_footprint()[:, 1], unit="degree")
        )
    except Exception:
        # fallback footprint
        pixel_boundaries = ([0, global_wcs.array_shape[1]], [0, global_wcs.array_shape[0]])

    # Convert to int for np.random.randint (world_to_pixel returns floats)
    max_x = max(1, int(np.floor(max(pixel_boundaries[0]))))
    max_y = max(1, int(np.floor(max(pixel_boundaries[1]))))

    # Generate random starting positions for each object within the global WCS frame
    xs = np.random.randint(0, max_x, n_objs_per_ic) + np.random.uniform(0, 1, n_objs_per_ic)
    ys = np.random.randint(0, max_y, n_objs_per_ic) + np.random.uniform(0, 1, n_objs_per_ic)

    # Get the velocity vectors for each trajectory
    vx_arr = np.array([t.vx for t in trjs])
    vy_arr = np.array([t.vy for t in trjs])

    # Add some jitter to the velocity vectors to simulate the fact that the
    # trajectories are not perfectly aligned with the ecliptic.
    unique_vx = np.unique([t.vx for t in candidates])
    unique_vy = np.unique([t.vy for t in candidates])
    dvx_arr = np.diff(unique_vx).mean() if len(unique_vx) > 1 else 0.0
    dvy_arr = np.diff(unique_vy).mean() if len(unique_vy) > 1 else 0.0
    if dvx_arr > 0:
        vx_arr += np.random.uniform(0, dvx_arr, n_objs_per_ic)
    if dvy_arr > 0:
        vy_arr += np.random.uniform(0, dvy_arr, n_objs_per_ic)

    # Ensure our obstimes are sorted for calculating the time steps for our synthetic trajectories
    # And since multiple images may have the same obstime, we need to make sure we don't have duplicate time steps
    obstimes = ic["mjd_mid"].copy()
    obstimes.sort()
    unique_obstimes = np.unique(obstimes)

    # Cumulative delta t from the first observation and calculate the positions of the objects
    # at each observation time.
    dts = unique_obstimes - unique_obstimes[0]
    xs = xs[:, None] + dts * vx_arr[:, None]
    ys = ys[:, None] + dts * vy_arr[:, None]

    # ra_orig/dec_orig are straight-line coords in the global WCS frame from the synthetic x and y positions
    sky_coords = global_wcs.pixel_to_world(xs, ys)
    ra_orig = sky_coords.ra.deg.ravel()
    dec_orig = sky_coords.dec.deg.ravel()

    if guess_distance is None:
        # Our default is to inject at the straight-line positions in the global WCS frame (no parallax correction)
        ra_for_injection = ra_orig
        dec_for_injection = dec_orig
    else:
        # Now we want to invert the parallax correction to inject at the correct positions in the orignal
        # exposures before they were resampled.
        loc = ic.get_observatory()
        if loc is None:
            raise ValueError("Observatory location not found in ImageCollection.")

        sky_coords_with_distance = SkyCoord(
            sky_coords.ra, sky_coords.dec, distance=guess_distance * u.au, frame="icrs"
        )
        invert_corrected_skycoords = kbmod.reprojection_utils.invert_correct_parallax_vectorized(
            sky_coords_with_distance, unique_obstimes, loc
        )

        # Inverse-corrected coords go into ra/dec (for VisitInjectTask on Butler exposures)
        ra_for_injection = invert_corrected_skycoords.ra.deg.ravel()
        dec_for_injection = invert_corrected_skycoords.dec.deg.ravel()

    obj_ids, mags, ts = [], [], []
    for i, x in enumerate(xs):
        obj_ids.extend([i] * len(x))
        mags.extend([np.random.uniform(mag_range[0], mag_range[1])] * len(x))
        ts.extend(unique_obstimes)

    catalog_dict = {
        "injection_id": np.arange(len(obj_ids)),
        "ra": ra_for_injection,
        "dec": dec_for_injection,
        "mag": mags,
        "guess_distance": [guess_distance] * len(obj_ids),
        "source_type": [source_type] * len(obj_ids),
        "obj_ids": obj_ids,
        "obstime": ts,
        "plot_x": xs.ravel(),
        "plot_y": ys.ravel(),
    }

    if guess_distance is not None:
        # "Straight-line" coords in the reflex-corrected global WCS (for plotting on resampled images)
        # This is convenient for recovering the synthetic object positions on the resampled images
        # but is not used for injection itself.
        catalog_dict[f"ra_{float(guess_distance)}"] = ra_orig
        catalog_dict[f"dec_{float(guess_distance)}"] = dec_orig

    return Table(catalog_dict)


def inject_sources_into_ic(ic, catalog, butler, inject_config=None):
    """
    Inject simulated moving objects directly into the exposures specified by an ImageCollection
    utilizing LSST pipelines. Note that this currently only works for `ButlerStandardizer` backed
    ImageCollections.

    This produces a new ImageCollection with the same structure as the input, but with injected
    sources in the image data with exposures modified by LSST's VisitInjectTask.

    Note that serializing the returned ImageCollection will not serialize the injected sources
    in the image data, instead it should be materialized as a `WorkUnit` via `ic.toWorkUnit()`
    in order to persist the injected sources.

    Parameters
    ----------
    ic : `kbmod.ImageCollection`
        Image collection to inject sources into.
    catalog : `astropy.table.Table`
        Catalog of sources to inject.
    butler : `dafButler.Butler`
        Butler to use for retrieving exposures.
    inject_config : `VisitInjectConfig`, optional
        Configuration for VisitInjectTask.

    Returns
    -------
    ic : `kbmod.ImageCollection`
        Re-built collection with injected sources from the catalog
    injected_cats : `astropy.table.Table`
        Restacked catalog from each call to LSST's VisitInjectTask
    """
    if not HAS_LSST:
        raise ImportError("LSST Science Pipelines must be installed to inject sources.")

    # Validate that the ImageCollection has the required columns for Butler-backed injection
    required_cols = ["dataId", "mjd_mid"]
    missing_cols = [col for col in required_cols if col not in ic.data.colnames]
    if missing_cols:
        raise ValueError(
            f"inject_sources_into_ic requires a Butler-backed ImageCollection with columns: "
            f"{required_cols}. Missing: {missing_cols}"
        )

    if inject_config is None:
        inject_config = VisitInjectConfig()
    inject_task = VisitInjectTask(config=inject_config)

    catalog = catalog.group_by("obstime")

    # Provide robust alignment handling
    references, exposures, injected_cats = [], [], []

    # Iterate through each exposure in the image collection for injection
    injected_exposure_cnt = 0
    for i in range(len(ic)):
        dataId, mjd_mid = ic.data["dataId"][i], ic.data["mjd_mid"][i]
        # Filter for all sources at our current timestep
        src_mask = catalog["obstime"] == mjd_mid
        srccat = catalog[src_mask]

        # Get the
        did = DatasetId(dataId)
        ref = butler.get_dataset(did, dimension_records=True)
        imdiff = butler.get(ref)

        if len(srccat) == 0:
            # If no sources are found for this timestep, append the original exposure and an empty catalog
            exposures.append(imdiff)
            injected_cats.append(
                Table(names=catalog.colnames, dtype=[catalog[c].dtype for c in catalog.colnames])
            )
            references.append(ref)
            continue

        try:
            # Run the injection task on the current exposure
            result = inject_task.run(
                injection_catalogs=srccat,
                input_exposure=imdiff,
                psf=imdiff.psf,
                photo_calib=imdiff.photoCalib,
                wcs=imdiff.wcs,
            )
            exposures.append(result.output_exposure)
            injected_cats.append(result.output_catalog)
            injected_exposure_cnt += 1
        except RuntimeError:
            # If no objects are rendered within bounds, append the original exposure and an empty catalog
            warnings.warn(
                f"Exposure {i}/{len(ic)} ({dataId}) had no objects successfully rendered within bounds."
            )
            exposures.append(imdiff)
            injected_cats.append(
                Table(names=catalog.colnames, dtype=[catalog[c].dtype for c in catalog.colnames])
            )
        references.append(ref)

    if injected_exposure_cnt == 0:
        warnings.warn("No objects were successfully rendered within bounds.")
    else:
        print(f"Successfully injected sources into {injected_exposure_cnt}/{len(ic)} exposures.")

    # Stack all the injected catalogs
    injected_cats = vstack(injected_cats)

    # Rebuild the standardizers with the new exposures
    standardizers = ic.get_standardizers(butler=butler)
    for std, ref, exp in zip(standardizers, references, exposures):
        std["std"].exp = exp
        std["std"].ref = ref

    # Deferred import to avoid circular dependency: image_collection imports injection
    from kbmod.image_collection import ImageCollection

    # Rebuild the ImageCollection from the new standardizers
    new_ic = ImageCollection.fromStandardizers([std["std"] for std in standardizers])
    return new_ic, injected_cats


def match_injection_results(
    catalog,
    results,
    guess_distance=None,
    sep_thresh=5.0,
    time_thresh_s=60.0,
    min_obs=3,
    matcher_name="injected_sources",
):
    """Match KBMOD search results against an injection catalog.

    This is a convenience wrapper around ``KnownObjsMatcher`` that adapts
    the injection catalog columns to the matcher's expected format and
    runs the match + recovery analysis in one call.

    The ``wcs`` and ``obstimes`` are pulled directly from the ``Results``
    object (``results.wcs`` and ``results.mjd_mid``).

    Parameters
    ----------
    catalog : `astropy.table.Table`, `str`, or `pathlib.Path`
        Injection catalog (astropy Table) or a path to a serialized file
        (``.parquet`` or ``.ecsv``).  Must contain ``obstime``, ``obj_ids``,
        and either ``ra_{guess_distance}``/``dec_{guess_distance}`` or
        ``ra``/``dec`` columns.
    results : `kbmod.Results` or `str`
        KBMOD search results (``Results`` object) or a path to a serialized
        results file.  Must have ``wcs`` and ``mjd_mid`` set (standard
        when created by ``run_search``).
    guess_distance : `float` or `None`, optional
        The guess distance in AU used during injection.  When provided the
        matcher uses the ``ra_{dist}``/``dec_{dist}`` columns which are the
        straight-line positions in the reflex-corrected global WCS — matching
        the coordinate frame of KBMOD results.  When ``None`` the
        ``guess_distance`` value is read from the catalog's
        ``guess_distance`` column.
    sep_thresh : `float`, optional
        Maximum spatial separation in arcseconds for a match. Default 5.0.
    time_thresh_s : `float`, optional
        Maximum time separation in seconds for a match. Default 60.0.
    min_obs : `int`, optional
        Minimum number of matching observations for a result to be
        considered a recovery. Default 3.
    matcher_name : `str`, optional
        Name used for the filter/matching column added to the results table.
        Default ``"injected_sources"``.

    Returns
    -------
    results : `kbmod.Results`
        The ``Results`` object with added columns:
        - ``{matcher_name}`` — per-result dict of matched object observations
        - ``recovered_{matcher_name}_min_obs_{min_obs}`` — list of recovered
          object names per result
    recovered : `set`
        Set of injected object names that were recovered.
    missed : `set`
        Set of injected object names that were not recovered.
    """
    # Load a serialized catalog if needed
    if isinstance(catalog, (str,)):
        import pathlib

        catalog_path = pathlib.Path(catalog)
        if catalog_path.suffix == ".parquet":
            import pandas as pd

            catalog = Table.from_pandas(pd.read_parquet(catalog_path))
        elif catalog_path.suffix == ".ecsv":
            catalog = Table.read(catalog_path, format="ascii.ecsv")
        else:
            catalog = Table.read(catalog_path)

    # Load a serialized KBMOD Results table if needed
    if isinstance(results, (str,)):
        results = Results.read_table(results)

    # Pull WCS and obstimes from results
    wcs = results.wcs
    obstimes = results.mjd_mid
    if wcs is None:
        raise ValueError(
            "Results object has no WCS. Ensure results were created by run_search or loaded with metadata."
        )
    if obstimes is None or len(obstimes) == 0:
        raise ValueError(
            "Results object has no mjd_mid. Ensure results were created by run_search or loaded with metadata."
        )

    # Determine guess_distance
    if guess_distance is None and "guess_distance" in catalog.colnames:
        gd_vals = catalog["guess_distance"]
        non_none = [v for v in gd_vals if v is not None]
        if len(non_none) > 0:
            guess_distance = float(non_none[0])

    # Determine ra/dec columns
    if guess_distance is not None:
        # Use ra_{dist}/dec_{dist} (straight-line in reflex-corrected global WCS)
        # since KBMOD results are in the same reflex-corrected frame.
        ra_col = f"ra_{float(guess_distance)}"
        dec_col = f"dec_{float(guess_distance)}"
        if ra_col not in catalog.colnames or dec_col not in catalog.colnames:
            warnings.warn(f"Columns {ra_col}/{dec_col} not found; falling back to ra/dec.")
            ra_col, dec_col = "ra", "dec"
    else:
        ra_col, dec_col = "ra", "dec"

    # Ensure obj_ids are strings (KnownObjsMatcher uses them as name keys)
    if "obj_ids" in catalog.colnames:
        catalog["obj_ids"] = [str(oid) for oid in catalog["obj_ids"]]

    # Create the matcher
    matcher = KnownObjsMatcher(
        table=catalog,
        obstimes=obstimes,
        matcher_name=matcher_name,
        sep_thresh=sep_thresh,
        time_thresh_s=time_thresh_s,
        mjd_col="obstime",
        ra_col=ra_col,
        dec_col=dec_col,
        name_col="obj_ids",
    )

    results = matcher.match(results, wcs)
    results = matcher.match_on_min_obs(results, min_obs=min_obs)

    match_col = matcher.match_min_obs_col(min_obs)
    recovered, missed = matcher.get_recovered_objects(results, match_col)

    n_total = len(set(catalog["obj_ids"]))
    print(
        f"Injection recovery: {len(recovered)}/{n_total} objects recovered "
        f'({len(missed)} missed) with min_obs={min_obs}, sep_thresh={sep_thresh}"'
    )

    return results, recovered, missed
