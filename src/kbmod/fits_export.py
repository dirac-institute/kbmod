"""FITS/CSV observation export for KBMOD results.

This module provides functions to export KBMOD search result trajectories
into combined FITS and/or CSV files compatible with the BulkFit orbit fitting tool.

The primary workflow:
1. Scan directories for ``*.search.parquet`` result files
2. Auto-derive ImageCollection from results file path for reference frame
3. Load results in chunks (memory-efficient for large files)
4. Extract per-observation RA/Dec from ``img_ra``/``img_dec`` columns,
   filtered by ``obs_valid`` masks
5. Build EXPNUM (1..N sorted by MJD) and TRACK_ID (integer from UUID) mappings
6. Write FITS file with DETECTIONS, EXPOSURES, and TRACK_ID_MAP extensions

Example
-------
>>> from kbmod.fits_export import export_results
>>> detections, exposures, trackid_map = export_results(
...     directories=["/path/to/results"],
...     output_path="/path/to/combined_observations.fits",
...     output_csv="/path/to/combined_observations.csv",
...     uuids=["abc123"],
... )
"""

import logging
from pathlib import Path

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, get_body_barycentric
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time
import astropy.units as u
import tqdm

from kbmod.results import Results

logger = logging.getLogger(__name__)


def parse_mpc80_line(line):
    """Parse a single MPC 80-column format line.

    Parameters
    ----------
    line : `str`
        A single line in MPC 80-column format.

    Returns
    -------
    obs : `dict` or `None`
        Dictionary with keys: designation, mjd, ra_deg, dec_deg, mag, band, obs_code.
        Returns None if line cannot be parsed.
    """
    import re

    # Pad line to at least 80 characters
    line = line.ljust(80)

    try:
        # Columns 1-12: Designation (strip discovery asterisk from column 13)
        designation = line[0:12].strip().rstrip("*")

        # Find date pattern: YYYY MM DD.ddddd (4-digit year, month, fractional day)
        # The date should be around columns 16-32 but may be shifted
        date_match = re.search(r"(\d{4})\s+(\d{2})\s+(\d{2}\.\d+)", line)
        if not date_match:
            return None

        year = int(date_match.group(1))
        month = int(date_match.group(2))
        day_frac = float(date_match.group(3))
        day = int(day_frac)
        frac = day_frac - day

        # Convert to MJD using astropy
        t = Time(f"{year}-{month:02d}-{day:02d}", format="iso", scale="utc")
        mjd = t.mjd + frac

        # Find RA pattern: HH MM SS.sss (after the date)
        # RA is in hours: 0-23 for hours, 0-59 for minutes, 0-59.xx for seconds
        date_end = date_match.end()
        ra_match = re.search(r"(\d{1,2})\s+(\d{2})\s+(\d{2}\.\d+)", line[date_end:])
        if not ra_match:
            return None

        ra_h = float(ra_match.group(1))
        ra_m = float(ra_match.group(2))
        ra_s = float(ra_match.group(3))
        ra_deg = (ra_h + ra_m / 60 + ra_s / 3600) * 15.0  # Convert hours to degrees

        # Find Dec pattern: sDD MM SS.ss (after RA, with optional sign)
        ra_end = date_end + ra_match.end()
        dec_match = re.search(r"([+-]?\d{1,2})\s+(\d{2})\s+(\d{2}\.\d+)", line[ra_end:])
        if not dec_match:
            return None

        dec_d = float(dec_match.group(1))
        dec_m = float(dec_match.group(2))
        dec_s = float(dec_match.group(3))
        sign = -1 if dec_d < 0 else 1
        dec_deg = sign * (abs(dec_d) + dec_m / 60 + dec_s / 3600)

        # Observatory code is last 3 characters
        obs_code = line[-3:].strip()

        # Magnitude - try to extract from around columns 66-71
        mag = None
        band = ""
        mag_match = re.search(r"(\d+\.\d)\s*([a-zA-Z]?)", line[55:72])
        if mag_match:
            mag = float(mag_match.group(1))
            band = mag_match.group(2) if mag_match.group(2) else ""

        return {
            "designation": designation,
            "mjd": mjd,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "mag": mag,
            "band": band,
            "obs_code": obs_code,
        }
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse MPC line: {line!r}, error: {e}")
        return None


def load_mpc80_file(filepath):
    """Load observations from an MPC 80-column format file.

    Parameters
    ----------
    filepath : `str` or `Path`
        Path to the MPC 80-column format file.

    Returns
    -------
    observations : `list` of `dict`
        List of observation dictionaries with keys: designation, mjd, ra_deg, dec_deg.
    """
    observations = []
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            obs = parse_mpc80_line(line)
            if obs is not None:
                observations.append(obs)

    logger.info(f"Loaded {len(observations)} observations from {filepath}")
    return observations


def load_existing_objects(directory):
    """Load all existing objects from MPC 80-column files in a directory.

    Parameters
    ----------
    directory : `str` or `Path`
        Path to directory containing MPC 80-column format files (*.txt).

    Returns
    -------
    existing_objects : `list` of `dict`
        List of objects, each with keys: name, observations (list of dicts).
    """
    directory = Path(directory)
    existing_objects = []

    # Find all .txt files
    mpc_files = sorted(directory.glob("*.txt"))
    logger.info(f"Found {len(mpc_files)} MPC files in {directory}")

    for mpc_file in mpc_files:
        observations = load_mpc80_file(mpc_file)
        if observations:
            # Use filename (without extension) as object name
            name = mpc_file.stem
            # Get designation from first observation
            designation = observations[0]["designation"] if observations else name
            existing_objects.append(
                {
                    "name": name,
                    "designation": designation,
                    "observations": observations,
                    "source_file": str(mpc_file),
                }
            )

    logger.info(f"Loaded {len(existing_objects)} existing objects")
    return existing_objects


def get_image_collection_path(results_path):
    """Derive ImageCollection path from results parquet file.

    The ImageCollection file is in the same directory with the same base name
    up to and including ``.collection``.

    Parameters
    ----------
    results_path : `str` or `Path`
        Path to the results parquet file.

    Returns
    -------
    ic_path : `Path` or `None`
        Path to the ImageCollection file, or None if cannot be derived.

    Examples
    --------
    >>> get_image_collection_path("468929_67.0_20X20_116_to_215.collection.wu.67.0.repro.search.parquet")
    PosixPath('468929_67.0_20X20_116_to_215.collection')
    """
    path = Path(results_path)
    name = path.name
    if ".collection" in name:
        # Strip everything after .collection
        base = name.split(".collection")[0] + ".collection"
        return path.parent / base
    return None


def load_image_collection_for_results(results_path):
    """Load ImageCollection and extract helio_guess_dist from metadata.

    Parameters
    ----------
    results_path : `str` or `Path`
        Path to the results parquet file.

    Returns
    -------
    ic : `ImageCollection` or `None`
        The loaded ImageCollection, or None if not found.
    helio_dist : `float` or `None`
        The heliocentric guess distance in AU from metadata.
    """
    from kbmod.image_collection import ImageCollection

    ic_path = get_image_collection_path(results_path)
    if ic_path is None or not ic_path.exists():
        logger.warning(f"ImageCollection not found for {results_path}")
        return None, None

    logger.info(f"Loading ImageCollection from {ic_path}")
    ic = ImageCollection.read(str(ic_path))
    # helio_guess_dist is unpacked as a column after reading
    helio_dist = None
    if "helio_guess_dist" in ic.data.colnames:
        helio_dist = float(ic.data["helio_guess_dist"][0])
    return ic, helio_dist


def compute_reference_frame_from_existing(existing_objects, observatory=None):
    """Compute RA0, DEC0, MJD0, X0, Y0, Z0 from existing objects.

    Reference frame is based on median RA/DEC of all existing object observations.

    Parameters
    ----------
    existing_objects : `list` of `dict`
        List of existing objects with 'observations' key.
    observatory : `EarthLocation` or `None`
        Observatory location. Defaults to Rubin if None.

    Returns
    -------
    ra0, dec0, mjd0, x0, y0, z0 : `float`
        Reference frame values for BulkFit header.
    """
    if observatory is None:
        observatory = EarthLocation(
            lat=-30.24463333 * u.deg,
            lon=-70.74941667 * u.deg,
            height=2662.75 * u.m,
        )

    # Collect all RA/DEC/MJD from existing objects
    all_ra = []
    all_dec = []
    all_mjd = []

    for obj in existing_objects:
        for obs in obj["observations"]:
            all_ra.append(obs["ra_deg"])
            all_dec.append(obs["dec_deg"])
            all_mjd.append(obs["mjd"])

    if not all_ra:
        raise ValueError("No observations in existing objects")

    # RA0, DEC0 = median of all existing object observations
    ra0 = float(np.median(all_ra))
    dec0 = float(np.median(all_dec))

    # MJD0 = first observation time
    mjd0 = float(np.min(all_mjd))

    # X0, Y0, Z0 = observatory barycentric position at MJD0
    obstime = Time(mjd0, format="mjd")
    earth_pos = get_body_barycentric("earth", obstime)

    # Get observatory offset in AU
    obs_gcrs = observatory.get_gcrs(obstime)
    obs_gcrs.representation_type = "cartesian"

    x0 = float((earth_pos.x + obs_gcrs.cartesian.x).to(u.AU).value)
    y0 = float((earth_pos.y + obs_gcrs.cartesian.y).to(u.AU).value)
    z0 = float((earth_pos.z + obs_gcrs.cartesian.z).to(u.AU).value)

    return ra0, dec0, mjd0, x0, y0, z0


def compute_reference_frame(image_collection, mjd_mid_array, observatory=None, barycentric_distance=None):
    """Compute RA0, DEC0, MJD0, X0, Y0, Z0 for BulkFit header.

    RA0/DEC0 come from the ImageCollection's global WCS center, with
    inverse reflex correction applied if barycentric_distance is provided.

    Parameters
    ----------
    image_collection : `ImageCollection`
        The ImageCollection to extract reference frame from.
    mjd_mid_array : `np.ndarray`
        Array of all MJD mid times across observations.
    observatory : `EarthLocation` or `None`
        Observatory location. Defaults to Rubin if None.
    barycentric_distance : `float` or `None`
        Barycentric distance in AU for inverse reflex correction.

    Returns
    -------
    ra0, dec0, mjd0, x0, y0, z0 : `float`
        Reference frame values for BulkFit header.
    """
    from kbmod.reprojection_utils import invert_correct_parallax_vectorized

    observatory = image_collection.get_observatory()
    if observatory is None:
        observatory = EarthLocation.of_site("Rubin")

    # MJD0 = first observation time (sorted)
    mjd0 = float(np.min(mjd_mid_array))

    # RA0, DEC0 from ImageCollection global WCS center
    ra0, dec0 = None, None
    global_wcs = image_collection.get_global_wcs(auto_fit=True)
    if global_wcs is not None and global_wcs.pixel_shape is not None:
        # Get center pixel
        cy, cx = global_wcs.pixel_shape[0] / 2, global_wcs.pixel_shape[1] / 2
        center_coord = global_wcs.pixel_to_world(cx, cy)

        # Apply inverse reflex correction if we have barycentric distance
        if barycentric_distance is not None:
            # The global WCS is in EBD (reflex-corrected) space
            # Invert to get observed ICRS coordinates
            center_coord_with_dist = SkyCoord(
                ra=center_coord.ra, dec=center_coord.dec, distance=barycentric_distance * u.AU
            )
            inverted = invert_correct_parallax_vectorized(
                coords=center_coord_with_dist,
                obstimes=Time(mjd0, format="mjd"),
                point_on_earth=observatory,
            )
            ra0 = float(inverted.ra.deg)
            dec0 = float(inverted.dec.deg)
        else:
            ra0 = float(center_coord.ra.deg)
            dec0 = float(center_coord.dec.deg)
    else:
        # Fallback: median of ImageCollection ra/dec columns
        ra0 = float(np.median(image_collection["ra"]))
        dec0 = float(np.median(image_collection["dec"]))

    # X0, Y0, Z0 = observatory barycentric position at MJD0
    obstime = Time(mjd0, format="mjd")
    earth_pos = get_body_barycentric("earth", obstime)

    # Get observatory offset in AU
    obs_gcrs = observatory.get_gcrs(obstime)
    obs_gcrs.representation_type = "cartesian"

    x0 = float((earth_pos.x + obs_gcrs.cartesian.x).to(u.AU).value)
    y0 = float((earth_pos.y + obs_gcrs.cartesian.y).to(u.AU).value)
    z0 = float((earth_pos.z + obs_gcrs.cartesian.z).to(u.AU).value)

    return ra0, dec0, mjd0, x0, y0, z0


def export_results(
    directories,
    output_path=None,
    output_csv=None,
    uuids=None,
    glob_pattern="*.search.parquet",
    chunk_size=100000,
    sigma_default=1.0,
    observatory=None,
    existing_objects_dir=None,
):
    """Export KBMOD result trajectories to combined FITS and/or CSV files.

    Scans one or more directories for result parquet files, processes
    them in memory-efficient chunks, and writes output files compatible
    with the BulkFit orbit fitting tool.

    When existing_objects_dir is provided, operates in "prepend mode":
    - Existing objects from MPC 80-column files get TRACK_IDs 1, 2, 3, ...
    - Each KBMOD result is duplicated N times (once per existing object)
    - Each duplicate has the existing object's observations prepended
    - Reference frame (RA0, DEC0) is based on median of existing objects

    The FITS output contains three extensions:
    - DETECTIONS: One row per valid observation with TRACK_ID, EXPNUM, RA, DEC, SIGMA
    - EXPOSURES: One row per unique exposure with EXPNUM, RA, DEC, MJD_MID
    - TRACK_ID_MAP: Mapping from TRACK_ID to UUID and source file

    The primary HDU header contains reference frame values:
    - RA0, DEC0: Central coordinates (from existing objects or ImageCollection)
    - MJD0: First observation time
    - X0, Y0, Z0: Observatory barycentric position at MJD0 (AU)

    Parameters
    ----------
    directories : `str`, `Path`, or `list`
        One or more directory paths to scan, or direct paths to
        individual ``.search.parquet`` files.
    output_path : `str` or `Path`, optional
        Path for the output FITS file. If None, no FITS is written.
    output_csv : `str` or `Path`, optional
        Path for the output CSV file (detections only). If None, no CSV.
    uuids : `set` or `list` of `str`, optional
        If provided, only export these UUIDs. If ``None``, export all.
    glob_pattern : `str`
        Glob pattern for finding result files within directories.
        Default: ``"*.search.parquet"``.
    chunk_size : `int`
        Number of rows to read at a time. Default: ``100000``.
    sigma_default : `float`
        Default astrometric uncertainty in arcsec. Default: ``1.0``.
    observatory : `EarthLocation` or `None`
        Observatory location. Defaults to Rubin if None.
    existing_objects_dir : `str` or `Path`, optional
        Directory containing MPC 80-column format files (*.txt) for existing
        objects. When provided, enables prepend mode.

    Returns
    -------
    detections_table : `astropy.table.Table`
        The detections table (TRACK_ID, EXPNUM, RA, DEC, SIGMA).
    exposures_table : `astropy.table.Table`
        The exposures table (EXPNUM, RA, DEC, MJD_MID).
    trackid_map_table : `astropy.table.Table`
        The track ID mapping table (TRACK_ID, UUID, SOURCE_FILE).
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]

    # Normalize UUID set for fast lookup
    uuid_set = set(uuids) if uuids is not None else None

    # Load existing objects if directory is provided
    existing_objects = []
    if existing_objects_dir is not None:
        existing_objects_dir = Path(existing_objects_dir)
        if existing_objects_dir.is_dir():
            existing_objects = load_existing_objects(existing_objects_dir)
            logger.info(f"Loaded {len(existing_objects)} existing objects for prepend mode")
        else:
            logger.warning(f"Existing objects directory not found: {existing_objects_dir}")

    # Collect paths: could be files directly or directories to scan
    result_paths = []
    for d in directories:
        d = Path(d)
        if d.is_file() and ".parquet" in d.suffix:
            result_paths.append(d)
        elif d.is_dir():
            result_paths.extend(sorted(d.rglob(glob_pattern)))
        else:
            logger.warning(f"Skipping {d}: not a file or directory")

    if not result_paths:
        logger.warning(f"No result files found matching '{glob_pattern}' in {directories}")
        empty_detections = Table(
            {
                "TRACK_ID": np.array([], dtype=np.int32),
                "EXPNUM": np.array([], dtype=np.int32),
                "RA": np.array([], dtype=np.float64),
                "DEC": np.array([], dtype=np.float64),
                "SIGMA": np.array([], dtype=np.float64),
            }
        )
        empty_exposures = Table(
            {
                "EXPNUM": np.array([], dtype=np.int32),
                "RA": np.array([], dtype=np.float64),
                "DEC": np.array([], dtype=np.float64),
                "MJD_MID": np.array([], dtype=np.float64),
            }
        )
        empty_trackid_map = Table(
            {
                "TRACK_ID": np.array([], dtype=np.int32),
                "UUID": np.array([], dtype=str),
                "SOURCE_FILE": np.array([], dtype=str),
            }
        )
        return empty_detections, empty_exposures, empty_trackid_map

    logger.info(f"Found {len(result_paths)} result file(s) to process")

    # First pass: collect all raw observation data
    # We need to collect everything first to build the MJD -> EXPNUM mapping
    raw_observations = []  # List of dicts with uuid, source_file, ra, dec, mjd
    track_info = []  # List of dicts with uuid, source_file for TRACK_ID_MAP

    # Also collect the first ImageCollection we find for reference frame
    first_image_collection = None
    first_helio_dist = None

    for result_path in tqdm.tqdm(result_paths, desc="Result files"):
        logger.info(f"Processing: {result_path}")
        source_file_str = str(result_path)

        # Try to load ImageCollection for this result
        if first_image_collection is None:
            ic, helio_dist = load_image_collection_for_results(result_path)
            if ic is not None:
                first_image_collection = ic
                first_helio_dist = helio_dist
                logger.info(f"Using ImageCollection from {result_path}, helio_dist={helio_dist}")

        try:
            for results_chunk in Results.read_table_chunks(str(result_path), chunk_size=chunk_size):
                table = results_chunk.table

                # Validate required columns
                if "img_ra" not in table.colnames or "img_dec" not in table.colnames:
                    logger.warning(
                        f"Skipping {result_path}: missing img_ra/img_dec columns. "
                        f"Available columns: {table.colnames}"
                    )
                    break
                if "obs_valid" not in table.colnames:
                    logger.warning(f"Skipping {result_path}: missing obs_valid column.")
                    break

                # Get MJD timestamps
                mjd_mid = None
                if results_chunk.mjd_mid is not None:
                    mjd_mid = np.asarray(results_chunk.mjd_mid)
                elif "mjd_mid" in table.meta:
                    mjd_mid = np.asarray(table.meta["mjd_mid"])
                elif "mjd_utc_mid" in table.meta:
                    mjd_mid = np.asarray(table.meta["mjd_utc_mid"])

                if mjd_mid is None:
                    logger.warning(f"Skipping {result_path}: no mjd_mid timestamps found.")
                    break

                # Process each row in the chunk
                for i in tqdm.tqdm(
                    range(len(table)),
                    desc=f"  {result_path.name}",
                    leave=False,
                ):
                    row = table[i]
                    row_uuid = str(row["uuid"])

                    # Skip if we have a specific UUID list and this isn't in it
                    if uuid_set is not None and row_uuid not in uuid_set:
                        continue

                    obs_valid = np.asarray(row["obs_valid"], dtype=bool)
                    valid_idx = np.where(obs_valid)[0]
                    n_valid = len(valid_idx)

                    # Add track info (for TRACK_ID_MAP)
                    track_info.append({"uuid": row_uuid, "source_file": source_file_str})

                    # Add observation rows if any valid
                    if n_valid > 0:
                        img_ra = np.asarray(row["img_ra"])
                        img_dec = np.asarray(row["img_dec"])

                        for idx in valid_idx:
                            raw_observations.append(
                                {
                                    "uuid": row_uuid,
                                    "source_file": source_file_str,
                                    "ra": float(img_ra[idx]),
                                    "dec": float(img_dec[idx]),
                                    "mjd": float(mjd_mid[idx]),
                                }
                            )

                        logger.debug(f"UUID {row_uuid}: extracted {n_valid} observations")

        except Exception as e:
            logger.error(f"Error processing {result_path}: {e}")
            raise

    logger.info(f"Total raw observations collected: {len(raw_observations)}")
    logger.info(f"Total tracks: {len(track_info)}")

    # Handle existing objects mode (prepend mode)
    if existing_objects:
        n_existing = len(existing_objects)
        logger.info(f"Prepend mode: {n_existing} existing objects")

        # Collect all MJDs from existing objects AND KBMOD results
        all_mjds_list = [obs["mjd"] for obs in raw_observations]
        for obj in existing_objects:
            for obs in obj["observations"]:
                all_mjds_list.append(obs["mjd"])

        all_mjds = np.array(all_mjds_list) if all_mjds_list else np.array([])
        unique_mjds = np.sort(np.unique(all_mjds)) if len(all_mjds) > 0 else np.array([])
        mjd_to_expnum = {mjd: i + 1 for i, mjd in enumerate(unique_mjds)}

        # Build unique UUIDs list and uuid->source_file mapping from KBMOD results
        unique_uuids = []
        seen_uuids = set()
        uuid_to_source_file = {}
        for t in track_info:
            if t["uuid"] not in seen_uuids:
                unique_uuids.append(t["uuid"])
                seen_uuids.add(t["uuid"])
                uuid_to_source_file[t["uuid"]] = t["source_file"]
        n_kbmod = len(unique_uuids)

        # Group KBMOD observations by UUID (do this once upfront)
        uuid_to_observations = {}
        for obs in raw_observations:
            uuid = obs["uuid"]
            if uuid not in uuid_to_observations:
                uuid_to_observations[uuid] = []
            uuid_to_observations[uuid].append(obs)

        # Pre-compute existing object observations as arrays for efficiency
        existing_obs_data = []
        for obj in existing_objects:
            obj_expnums = np.array([mjd_to_expnum[obs["mjd"]] for obs in obj["observations"]], dtype=np.int32)
            obj_ras = np.array([obs["ra_deg"] for obs in obj["observations"]], dtype=np.float64)
            obj_decs = np.array([obs["dec_deg"] for obs in obj["observations"]], dtype=np.float64)
            existing_obs_data.append({
                "designation": obj["designation"],
                "source_file": obj["source_file"],
                "expnums": obj_expnums,
                "ras": obj_ras,
                "decs": obj_decs,
                "n_obs": len(obj["observations"]),
            })

        # Calculate total sizes for pre-allocation
        total_existing_standalone = sum(e["n_obs"] for e in existing_obs_data)
        total_prepended_tracks = n_kbmod * n_existing
        total_kbmod_obs = len(raw_observations)
        # Each prepended track has: existing obs + KBMOD obs for that UUID
        # Estimate: avg obs per KBMOD track * n_existing + existing obs * n_kbmod
        avg_obs_per_kbmod = total_kbmod_obs / n_kbmod if n_kbmod > 0 else 0
        est_prepended_detections = int(total_prepended_tracks * (avg_obs_per_kbmod + sum(e["n_obs"] for e in existing_obs_data) / n_existing))

        logger.info(f"Estimated detections: {total_existing_standalone} (existing) + ~{est_prepended_detections} (prepended)")

        # TRACK_ID assignment:
        # 1..N_existing: existing objects alone
        # N_existing+1..N_existing+N_kbmod*N_existing: KBMOD results prepended with existing objects

        # Build arrays using numpy for efficiency
        trackid_map_track_ids = []
        trackid_map_uuids = []
        trackid_map_sources = []

        det_track_ids = []
        det_expnums = []
        det_ras = []
        det_decs = []

        current_track_id = 1

        # First, add existing objects as standalone tracks (TRACK_ID 1, 2, 3, ...)
        logger.info("Adding existing objects as standalone tracks...")
        for i, eod in enumerate(existing_obs_data):
            track_id = current_track_id
            current_track_id += 1

            trackid_map_track_ids.append(track_id)
            trackid_map_uuids.append(eod["designation"])
            trackid_map_sources.append(eod["source_file"])

            n = eod["n_obs"]
            det_track_ids.extend([track_id] * n)
            det_expnums.extend(eod["expnums"].tolist())
            det_ras.extend(eod["ras"].tolist())
            det_decs.extend(eod["decs"].tolist())

        # Now, for each KBMOD result, create N_existing copies prepended with each existing object
        logger.info(f"Creating {total_prepended_tracks} prepended track combinations...")
        for idx, uuid in enumerate(tqdm.tqdm(unique_uuids, desc="Prepending tracks")):
            kbmod_obs_list = uuid_to_observations.get(uuid, [])
            source_file = uuid_to_source_file.get(uuid, "")

            # Pre-extract KBMOD obs data for this UUID
            if kbmod_obs_list:
                kbmod_expnums = [mjd_to_expnum[obs["mjd"]] for obs in kbmod_obs_list]
                kbmod_ras = [obs["ra"] for obs in kbmod_obs_list]
                kbmod_decs = [obs["dec"] for obs in kbmod_obs_list]
                n_kbmod_obs = len(kbmod_obs_list)
            else:
                kbmod_expnums = []
                kbmod_ras = []
                kbmod_decs = []
                n_kbmod_obs = 0

            for eod in existing_obs_data:
                track_id = current_track_id
                current_track_id += 1

                # TRACK_ID_MAP entry
                trackid_map_track_ids.append(track_id)
                trackid_map_uuids.append(f"{eod['designation']}+{uuid}")
                trackid_map_sources.append(f"{eod['source_file']}+{source_file}")

                # Existing object observations first
                n_existing_obs = eod["n_obs"]
                det_track_ids.extend([track_id] * n_existing_obs)
                det_expnums.extend(eod["expnums"].tolist())
                det_ras.extend(eod["ras"].tolist())
                det_decs.extend(eod["decs"].tolist())

                # Then KBMOD observations
                det_track_ids.extend([track_id] * n_kbmod_obs)
                det_expnums.extend(kbmod_expnums)
                det_ras.extend(kbmod_ras)
                det_decs.extend(kbmod_decs)

        logger.info(f"Created {current_track_id - 1} total tracks "
                   f"({n_existing} existing + {n_kbmod * n_existing} prepended)")

        # Convert to dict format for table creation
        detection_rows = None  # Signal to use arrays directly
        trackid_map_rows = None

    else:
        # Normal mode (no existing objects)
        # Build UUID -> TRACK_ID mapping (contiguous integers starting at 1)
        unique_uuids = []
        seen_uuids = set()
        for t in track_info:
            if t["uuid"] not in seen_uuids:
                unique_uuids.append(t["uuid"])
                seen_uuids.add(t["uuid"])
        uuid_to_trackid = {uuid: i + 1 for i, uuid in enumerate(unique_uuids)}

        # Build MJD -> EXPNUM mapping (contiguous integers starting at 1, sorted by time)
        all_mjds = np.array([obs["mjd"] for obs in raw_observations]) if raw_observations else np.array([])
        unique_mjds = np.sort(np.unique(all_mjds)) if len(all_mjds) > 0 else np.array([])
        mjd_to_expnum = {mjd: i + 1 for i, mjd in enumerate(unique_mjds)}

        # Build TRACK_ID_MAP rows
        trackid_map_rows = []
        for t in track_info:
            trackid_map_rows.append({
                "TRACK_ID": uuid_to_trackid[t["uuid"]],
                "UUID": t["uuid"],
                "SOURCE_FILE": t["source_file"],
            })

        # Build detection rows
        detection_rows = []
        for obs in raw_observations:
            detection_rows.append({
                "TRACK_ID": uuid_to_trackid[obs["uuid"]],
                "EXPNUM": mjd_to_expnum[obs["mjd"]],
                "RA": obs["ra"],
                "DEC": obs["dec"],
                "SIGMA": sigma_default,
            })

    # Create tables from rows or arrays
    if existing_objects:
        # Use array-based approach (faster for large datasets)
        logger.info("Building tables from arrays...")
        if trackid_map_track_ids:
            trackid_map_table = Table({
                "TRACK_ID": np.array(trackid_map_track_ids, dtype=np.int32),
                "UUID": np.array(trackid_map_uuids, dtype=str),
                "SOURCE_FILE": np.array(trackid_map_sources, dtype=str),
            })
        else:
            trackid_map_table = Table({
                "TRACK_ID": np.array([], dtype=np.int32),
                "UUID": np.array([], dtype=str),
                "SOURCE_FILE": np.array([], dtype=str),
            })

        if det_track_ids:
            detections_table = Table({
                "TRACK_ID": np.array(det_track_ids, dtype=np.int32),
                "EXPNUM": np.array(det_expnums, dtype=np.int32),
                "RA": np.array(det_ras, dtype=np.float64),
                "DEC": np.array(det_decs, dtype=np.float64),
                "SIGMA": np.full(len(det_track_ids), sigma_default, dtype=np.float64),
            })
        else:
            detections_table = Table({
                "TRACK_ID": np.array([], dtype=np.int32),
                "EXPNUM": np.array([], dtype=np.int32),
                "RA": np.array([], dtype=np.float64),
                "DEC": np.array([], dtype=np.float64),
                "SIGMA": np.array([], dtype=np.float64),
            })
    else:
        # Use dict-based approach (original method for non-prepend mode)
        if trackid_map_rows:
            trackid_map_table = Table(rows=trackid_map_rows)
        else:
            trackid_map_table = Table({
                "TRACK_ID": np.array([], dtype=np.int32),
                "UUID": np.array([], dtype=str),
                "SOURCE_FILE": np.array([], dtype=str),
            })

        if detection_rows:
            detections_table = Table(rows=detection_rows)
        else:
            detections_table = Table({
                "TRACK_ID": np.array([], dtype=np.int32),
                "EXPNUM": np.array([], dtype=np.int32),
                "RA": np.array([], dtype=np.float64),
                "DEC": np.array([], dtype=np.float64),
                "SIGMA": np.array([], dtype=np.float64),
            })

    # Build EXPOSURES table from all observations (both existing and KBMOD)
    all_observations_for_exposures = list(raw_observations)
    if existing_objects:
        for obj in existing_objects:
            for obs in obj["observations"]:
                all_observations_for_exposures.append({
                    "ra": obs["ra_deg"],
                    "dec": obs["dec_deg"],
                    "mjd": obs["mjd"],
                })

    all_mjds_for_exp = np.array([obs["mjd"] for obs in all_observations_for_exposures]) if all_observations_for_exposures else np.array([])
    unique_mjds = np.sort(np.unique(all_mjds_for_exp)) if len(all_mjds_for_exp) > 0 else np.array([])
    mjd_to_expnum = {mjd: i + 1 for i, mjd in enumerate(unique_mjds)}

    if len(unique_mjds) > 0:
        exposure_rows = []
        for mjd in unique_mjds:
            expnum = mjd_to_expnum[mjd]
            obs_at_mjd = [obs for obs in all_observations_for_exposures if obs["mjd"] == mjd]
            if obs_at_mjd:
                ras = [obs["ra"] for obs in obs_at_mjd]
                decs = [obs["dec"] for obs in obs_at_mjd]
                central_ra = float(np.median(ras))
                central_dec = float(np.median(decs))
            else:
                central_ra = 0.0
                central_dec = 0.0

            exposure_rows.append({
                "EXPNUM": expnum,
                "RA": central_ra,
                "DEC": central_dec,
                "MJD_MID": mjd,
            })
        exposures_table = Table(rows=exposure_rows)
    else:
        exposures_table = Table({
            "EXPNUM": np.array([], dtype=np.int32),
            "RA": np.array([], dtype=np.float64),
            "DEC": np.array([], dtype=np.float64),
            "MJD_MID": np.array([], dtype=np.float64),
        })

    logger.info(f"Total detections: {len(detections_table)}")
    logger.info(f"Total exposures: {len(exposures_table)}")
    logger.info(f"Total track IDs: {len(trackid_map_table)}")

    # Compute reference frame values
    ra0, dec0, mjd0, x0, y0, z0 = None, None, None, None, None, None

    if existing_objects:
        # Use existing objects for reference frame (median RA/DEC)
        try:
            ra0, dec0, mjd0, x0, y0, z0 = compute_reference_frame_from_existing(
                existing_objects, observatory=observatory
            )
            logger.info(f"Computed reference frame from existing objects: RA0={ra0:.6f}, DEC0={dec0:.6f}, MJD0={mjd0:.6f}")
            logger.info(f"  X0={x0:.6f}, Y0={y0:.6f}, Z0={z0:.6f} AU")
        except Exception as e:
            logger.warning(f"Failed to compute reference frame from existing objects: {e}")
    elif first_image_collection is not None and len(unique_mjds) > 0:
        # Use ImageCollection for reference frame
        try:
            ra0, dec0, mjd0, x0, y0, z0 = compute_reference_frame(
                first_image_collection,
                unique_mjds,
                observatory=observatory,
                barycentric_distance=first_helio_dist,
            )
            logger.info(f"Computed reference frame: RA0={ra0:.6f}, DEC0={dec0:.6f}, MJD0={mjd0:.6f}")
            logger.info(f"  X0={x0:.6f}, Y0={y0:.6f}, Z0={z0:.6f} AU")
        except Exception as e:
            logger.warning(f"Failed to compute reference frame: {e}")

    # Write FITS file
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["COMMENT"] = "KBMOD BulkFit-compatible observations export"
        primary_hdu.header["N_DET"] = len(detections_table)
        primary_hdu.header["N_EXP"] = len(exposures_table)
        primary_hdu.header["N_TRACK"] = len(trackid_map_table)
        primary_hdu.header["SIGMA"] = (sigma_default, "Default astrometric uncertainty (arcsec)")

        # Add reference frame values to header
        if ra0 is not None:
            primary_hdu.header["RA0"] = (ra0, "Reference RA (degrees)")
        if dec0 is not None:
            primary_hdu.header["DEC0"] = (dec0, "Reference DEC (degrees)")
        if mjd0 is not None:
            primary_hdu.header["MJD0"] = (mjd0, "Reference MJD (first observation)")
        if x0 is not None:
            primary_hdu.header["X0"] = (x0, "Observatory X barycentric (AU)")
        if y0 is not None:
            primary_hdu.header["Y0"] = (y0, "Observatory Y barycentric (AU)")
        if z0 is not None:
            primary_hdu.header["Z0"] = (z0, "Observatory Z barycentric (AU)")
        if first_helio_dist is not None:
            primary_hdu.header["HELIODST"] = (first_helio_dist, "Heliocentric guess distance (AU)")

        det_hdu = fits.BinTableHDU(detections_table, name="DETECTIONS")
        det_hdu.header["EXTNAME"] = "DETECTIONS"
        det_hdu.header["COMMENT"] = "Per-detection data for BulkFit"

        exp_hdu = fits.BinTableHDU(exposures_table, name="EXPOSURES")
        exp_hdu.header["EXTNAME"] = "EXPOSURES"
        exp_hdu.header["COMMENT"] = "Per-exposure data for AddObservatory"

        map_hdu = fits.BinTableHDU(trackid_map_table, name="TRACK_ID_MAP")
        map_hdu.header["EXTNAME"] = "TRACK_ID_MAP"
        map_hdu.header["COMMENT"] = "TRACK_ID to UUID mapping"

        hdu_list = fits.HDUList([primary_hdu, det_hdu, exp_hdu, map_hdu])
        hdu_list.writeto(str(output_path), overwrite=True)
        logger.info(f"Wrote FITS file: {output_path}")

    # Write CSV file (detections only)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        detections_table.write(str(output_csv), format="csv", overwrite=True)
        logger.info(f"Wrote CSV file: {output_csv}")

    return detections_table, exposures_table, trackid_map_table


def read_fits_export(fits_path):
    """Read back the tables from a FITS export file.

    Parameters
    ----------
    fits_path : `str` or `Path`
        Path to the FITS file created by ``export_results``.

    Returns
    -------
    detections_table : `astropy.table.Table`
        The detections table.
    exposures_table : `astropy.table.Table`
        The exposures table.
    trackid_map_table : `astropy.table.Table`
        The track ID mapping table.
    """
    with fits.open(fits_path) as hdul:
        detections_table = Table.read(hdul["DETECTIONS"])
        exposures_table = Table.read(hdul["EXPOSURES"])
        trackid_map_table = Table.read(hdul["TRACK_ID_MAP"])
    return detections_table, exposures_table, trackid_map_table
