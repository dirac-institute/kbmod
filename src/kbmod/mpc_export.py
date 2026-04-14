"""MPC 80-column observation format export for KBMOD results.

This module provides functions to convert KBMOD search result trajectories
into Minor Planet Center (MPC) 80-column formatted observation files,
suitable for orbit fitting with tools like pyOrbfit.

The primary workflow:
1. Scan directories for ``*.search.parquet`` result files
2. Load results in chunks (memory-efficient for large files)
3. Extract per-observation RA/Dec from ``img_ra``/``img_dec`` columns,
   filtered by ``obs_valid`` masks
4. Format each valid observation as an MPC 80-column line
5. Write one ``.txt`` file per UUID and a manifest table

Example
-------
>>> from kbmod.mpc_export import export_results_to_mpc_files
>>> manifest = export_results_to_mpc_files(
...     directories=["/path/to/results"],
...     output_dir="/path/to/mpc_output",
...     uuids=["abc123"],
...     observatory="X05",
... )
"""

import logging
from math import copysign
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.time import Time
import tqdm

from kbmod.results import Results

logger = logging.getLogger(__name__)


def _deg_to_hms(ra_deg):
    """Convert RA in degrees to (hours, minutes, seconds) tuple."""
    ra_h = ra_deg / 15.0
    h = int(ra_h)
    remainder = (ra_h - h) * 60.0
    m = int(remainder)
    s = (remainder - m) * 60.0
    return h, m, s


def _deg_to_dms(dec_deg):
    """Convert Dec in degrees to (sign, degrees, arcmin, arcsec) tuple.

    Returns
    -------
    sign : int
        +1 or -1
    d : int
        Absolute degrees
    m : int
        Absolute arcminutes
    s : float
        Absolute arcseconds
    """
    sign = 1 if dec_deg >= 0 else -1
    dec_abs = abs(dec_deg)
    d = int(dec_abs)
    remainder = (dec_abs - d) * 60.0
    m = int(remainder)
    s = (remainder - m) * 60.0
    return sign, d, m, s


def format_mpc_line(ra_deg, dec_deg, mjd, designation, observatory="X05"):
    """Format a single observation as an MPC 80-column line.

    Parameters
    ----------
    ra_deg : `float`
        Right ascension in degrees.
    dec_deg : `float`
        Declination in degrees.
    mjd : `float`
        Modified Julian Date of the observation (UTC).
    designation : `str`
        Provisional designation string (up to 7 characters, will be
        placed in columns 6-12 of the MPC line).
    observatory : `str`
        Three character observatory code. Default is ``"X05"`` (Rubin).

    Returns
    -------
    line : `str`
        An 80-character MPC formatted observation line.
    """
    t = Time(mjd, format="mjd", scale="utc")
    mjd_frac = mjd % 1.0

    h, m, s = _deg_to_hms(ra_deg)
    sign, dd, dm, ds = _deg_to_dms(dec_deg)

    # Date: "YYYY MM DD.ddddd"
    date_str = "%4i %02i %08.5f" % (t.datetime.year, t.datetime.month, t.datetime.day + mjd_frac)

    # RA: "HH MM SS.sss"
    ra_str = "%02i %02i %06.3f" % (h, m, s)

    # Dec: "+DD MM SS.ss" — handle zero-degree sign correctly
    if dd != 0:
        dec_str = "%+03i %02i %05.2f" % (sign * dd, dm, ds)
    else:
        dec_d_str = "-00" if sign == -1 else "+00"
        dec_str = "%s %02i %05.2f" % (dec_d_str, dm, ds)

    # MPC 80-column format (see 80_col.md reference):
    # Cols  1- 5: (5) blank or packed number
    # Cols  6-12: (7) provisional designation
    # Col  13   : (1) discovery asterisk (blank)
    # Col  14   : (1) note 1 (blank)
    # Col  15   : (1) note 2 — 'C' for CCD
    # Cols 16-32: (17) date of observation
    # Cols 33-44: (12) RA (J2000.0)
    # Cols 45-56: (12) Dec (J2000.0)
    # Cols 57-77: (21) blank (magnitude/band not available)
    # Cols 78-80: (3) observatory code
    line = "     %-7s  C%s %s%s                     %s" % (
        designation[:7],
        date_str,
        ra_str,
        dec_str,
        observatory[:3],
    )

    assert len(line) == 80, f"MPC line length is {len(line)}, expected 80: {repr(line)}"
    return line


def _format_mpc_lines_vectorized(ra_deg_arr, dec_deg_arr, mjd_arr, designation, observatory="X05"):
    """Vectorized formatting of multiple observations into MPC lines.

    Converts arrays of RA/Dec/MJD into MPC 80-column strings efficiently
    by batching the astropy Time conversion and using pure arithmetic
    for sexagesimal conversion.

    Parameters
    ----------
    ra_deg_arr : `numpy.ndarray`
        Array of RA values in degrees.
    dec_deg_arr : `numpy.ndarray`
        Array of Dec values in degrees.
    mjd_arr : `numpy.ndarray`
        Array of MJD values (UTC).
    designation : `str`
        Provisional designation (up to 7 chars).
    observatory : `str`
        Observatory code (3 chars).

    Returns
    -------
    lines : `list` of `str`
        MPC 80-column formatted lines.
    """
    if len(ra_deg_arr) == 0:
        return []

    # Batch convert MJD to datetime
    times = Time(mjd_arr, format="mjd", scale="utc")
    mjd_fracs = mjd_arr % 1.0

    # Vectorized RA -> HMS
    ra_h_float = ra_deg_arr / 15.0
    ra_h = ra_h_float.astype(int)
    ra_rem = (ra_h_float - ra_h) * 60.0
    ra_m = ra_rem.astype(int)
    ra_s = (ra_rem - ra_m) * 60.0

    # Vectorized Dec -> DMS
    dec_sign = np.where(dec_deg_arr >= 0, 1, -1)
    dec_abs = np.abs(dec_deg_arr)
    dec_d = dec_abs.astype(int)
    dec_rem = (dec_abs - dec_d) * 60.0
    dec_m = dec_rem.astype(int)
    dec_s = (dec_rem - dec_m) * 60.0

    desig = designation[:7]
    obs = observatory[:3]

    lines = []
    for i in range(len(ra_deg_arr)):
        dt = times[i].datetime
        date_str = "%4i %02i %08.5f" % (dt.year, dt.month, dt.day + mjd_fracs[i])
        ra_str = "%02i %02i %06.3f" % (ra_h[i], ra_m[i], ra_s[i])

        d_val = int(dec_d[i])
        if d_val != 0:
            dec_str = "%+03i %02i %05.2f" % (dec_sign[i] * d_val, dec_m[i], dec_s[i])
        else:
            dec_d_str = "-00" if dec_sign[i] == -1 else "+00"
            dec_str = "%s %02i %05.2f" % (dec_d_str, dec_m[i], dec_s[i])

        line = "     %-7s  C%s %s%s                     %s" % (
            desig,
            date_str,
            ra_str,
            dec_str,
            obs,
        )
        lines.append(line)

    return lines


def format_result_to_mpc(row, mjd_mid, observatory="X05"):
    """Convert a single KBMOD result row to MPC 80-column lines.

    Parameters
    ----------
    row : `dict`-like
        A single result row containing at minimum ``uuid``, ``img_ra``,
        ``img_dec``, and ``obs_valid``.
    mjd_mid : `numpy.ndarray`
        Array of observation times (MJD UTC) for all images.
    observatory : `str`
        Three character observatory code. Default is ``"X05"``.

    Returns
    -------
    mpc_lines : `list` of `str`
        List of MPC 80-column formatted observation strings.
    """
    obs_valid = np.asarray(row["obs_valid"], dtype=bool)
    img_ra = np.asarray(row["img_ra"])
    img_dec = np.asarray(row["img_dec"])
    uuid_str = str(row["uuid"])

    # Use first 7 characters of the UUID as the provisional designation
    designation = uuid_str[:7]

    valid_idx = np.where(obs_valid)[0]
    if len(valid_idx) == 0:
        return []

    return _format_mpc_lines_vectorized(
        img_ra[valid_idx],
        img_dec[valid_idx],
        mjd_mid[valid_idx],
        designation,
        observatory,
    )


def export_results_to_mpc_files(
    directories,
    output_dir,
    uuids=None,
    glob_pattern="*.search.parquet",
    observatory="X05",
    chunk_size=100000,
):
    """Export KBMOD result trajectories to MPC 80-column files.

    Scans one or more directories for result parquet files, processes
    them in memory-efficient chunks, and writes one MPC ``.txt`` file
    per UUID. Also produces a manifest table mapping UUIDs back to
    their source result files.

    Parameters
    ----------
    directories : `str`, `Path`, or `list`
        One or more directory paths to scan, or direct paths to
        individual ``.search.parquet`` files.
    output_dir : `str` or `Path`
        Directory where MPC files and the manifest will be written.
    uuids : `set` or `list` of `str`, optional
        If provided, only export these UUIDs. If ``None``, export all.
    glob_pattern : `str`
        Glob pattern for finding result files within directories.
        Default: ``"*.search.parquet"``.
    observatory : `str`
        Three character observatory code. Default: ``"X05"``.
    chunk_size : `int`
        Number of rows to read at a time. Default: ``100000``.

    Returns
    -------
    manifest : `astropy.table.Table`
        A table with columns ``uuid``, ``search_file``, ``mpc_file``,
        ``n_obs`` tracking what was exported.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(directories, (str, Path)):
        directories = [directories]

    # Normalize UUID set for fast lookup
    uuid_set = set(uuids) if uuids is not None else None

    manifest_rows = {
        "uuid": [],
        "search_file": [],
        "mpc_file": [],
        "n_obs": [],
    }

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
        return Table(manifest_rows)

    logger.info(f"Found {len(result_paths)} result file(s) to process")

    for result_path in tqdm.tqdm(result_paths, desc="Result files"):
        logger.info(f"Processing: {result_path}")

        try:
            for results_chunk in Results.read_table_chunks(str(result_path), chunk_size=chunk_size):
                # We need img_ra, img_dec, obs_valid columns and mjd_mid
                table = results_chunk.table
                if "img_ra" not in table.colnames or "img_dec" not in table.colnames:
                    logger.warning(
                        f"Skipping {result_path}: missing img_ra/img_dec columns. "
                        f"Available columns: {table.colnames}"
                    )
                    break
                if "obs_valid" not in table.colnames:
                    logger.warning(f"Skipping {result_path}: missing obs_valid column.")
                    break

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
                    row_uuid = str(table["uuid"][i])

                    # Skip if we have a specific UUID list and this isn't in it
                    if uuid_set is not None and row_uuid not in uuid_set:
                        continue

                    mpc_lines = format_result_to_mpc(table[i], mjd_mid, observatory=observatory)

                    if not mpc_lines:
                        logger.debug(f"UUID {row_uuid}: no valid observations, skipping")
                        continue

                    # Write MPC file
                    mpc_filename = f"{row_uuid}.txt"
                    mpc_filepath = output_dir / mpc_filename
                    with open(mpc_filepath, "w") as f:
                        for line in mpc_lines:
                            f.write(line + "\n")

                    manifest_rows["uuid"].append(row_uuid)
                    manifest_rows["search_file"].append(str(result_path))
                    manifest_rows["mpc_file"].append(str(mpc_filepath))
                    manifest_rows["n_obs"].append(len(mpc_lines))

                    logger.debug(f"UUID {row_uuid}: wrote {len(mpc_lines)} observations to {mpc_filepath}")

        except Exception as e:
            logger.error(f"Error processing {result_path}: {e}")

    manifest = Table(manifest_rows)

    # Write the manifest
    manifest_path = output_dir / "mpc_export_manifest.parquet"
    manifest.write(str(manifest_path), overwrite=True)
    logger.info(f"Exported {len(manifest)} MPC files. Manifest: {manifest_path}")

    return manifest
