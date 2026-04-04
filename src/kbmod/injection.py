import warnings
import numpy as np

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import kbmod.trajectory_generator
import kbmod.wcs_utils
import kbmod.reprojection_utils

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
    """Generate an injection catalog for an ImageCollection intended which can be consumed by `inject_sources_into_ic`.

    Incorporates sub-pixel and sub-search-velocity-resolution jitter, and handles coordinate propagation
    keeping objects moving in straight lines inside global WCS pixel coordinates.

    If no heliocentric guess distance is provided, the catalog will provide the coordinates in the original
    reference frame defined by `global_wcs`, which is presumed to not be reflex-corrected.

    If a guess distance is provided, the catalog will provide the coordinates in EBD which will presumed to be
    defined by `global_wcs` reflex-corrected at the specified guess distance. The injection coordinates will be
    the inverse parallax-corrected and and should no longer necessarily appear as straight "lines" until after
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

    max_x = max(pixel_boundaries[0])
    max_y = max(pixel_boundaries[1])

    # Generate random starting positions for each object within the global WCS frame
    xs = np.random.randint(0, max_x, n_objs_per_ic) + np.random.uniform(0, 1, n_objs_per_ic)
    ys = np.random.randint(0, max_y, n_objs_per_ic) + np.random.uniform(0, 1, n_objs_per_ic)

    # Get the velocity vectors for each trajectory
    vx_arr = np.array([t.vx for t in trjs])
    vy_arr = np.array([t.vy for t in trjs])

    # Add some jitter to the velocity vectors to simulate the fact that the
    # trajectories are not perfectly aligned with the ecliptic.
    dvx_arr = np.diff(np.unique([t.vx for t in candidates])).mean() if len(candidates) > 1 else 0.0
    dvy_arr = np.diff(np.unique([t.vy for t in candidates])).mean() if len(candidates) > 1 else 0.0
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
    utilizing LSST pipelines. NOte that this currently only works for `ButlerStandardizer` backed
    ImageCollections.

    This produces a new ImageCollection with the same structure as the input, but with injected
    sources in the image data with exposures modified by LSST's VisitInjectTask.

    Note that serializaing the returned ImageCollection will not serialize the injected sources
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

    if inject_config is None:
        inject_config = VisitInjectConfig()
    inject_task = VisitInjectTask(config=inject_config)

    catalog = catalog.group_by("obstime")

    # Provide robust alignment handling
    references, exposures, injected_cats = [], [], []

    for idd, obs_t in zip(ic.data["dataId"], ic.data["mjd_mid"]):
        # Exact matching for MJD
        src_mask = catalog["obstime"] == obs_t
        srccat = catalog[src_mask]

        did = DatasetId(idd)
        ref = butler.get_dataset(did, dimension_records=True)
        imdiff = butler.get(ref)

        if len(srccat) == 0:
            exposures.append(imdiff)
            injected_cats.append(
                Table(names=catalog.colnames, dtype=[catalog[c].dtype for c in catalog.colnames])
            )
            references.append(ref)
            continue

        try:
            result = inject_task.run(
                injection_catalogs=srccat,
                input_exposure=imdiff,
                psf=imdiff.psf,
                photo_calib=imdiff.photoCalib,
                wcs=imdiff.wcs,
            )
            exposures.append(result.output_exposure)
            injected_cats.append(result.output_catalog)
        except RuntimeError:
            warnings.warn(f"Exposure {idd} had no objects rendering within bounds.")
            exposures.append(imdiff)
            injected_cats.append(
                Table(names=catalog.colnames, dtype=[catalog[c].dtype for c in catalog.colnames])
            )

        references.append(ref)

    injected_cats = vstack(injected_cats)

    standardizers = ic.get_standardizers(butler=butler)
    if len(standardizers) != len(ic):
        raise RuntimeError("Recovered standardizers do not match IC length.")

    for std, ref, exp in zip(standardizers, references, exposures):
        std["std"].exp = exp
        std["std"].ref = ref

    # Deferred import to avoid circular dependency: image_collection imports injection
    from kbmod.image_collection import ImageCollection

    new_ic = ImageCollection.fromStandardizers([std["std"] for std in standardizers])
    return new_ic, injected_cats
