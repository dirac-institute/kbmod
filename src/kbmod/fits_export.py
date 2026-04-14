"""FITS/CSV observation export for KBMOD results.

This module provides functions to export KBMOD search result trajectories
into combined FITS and/or CSV files with per-observation rows, suitable for
downstream analysis and orbit fitting.

The primary workflow:
1. Scan directories for ``*.search.parquet`` result files
2. Load results in chunks (memory-efficient for large files)
3. Extract per-observation RA/Dec from ``img_ra``/``img_dec`` columns,
   filtered by ``obs_valid`` masks
4. Combine all observations into a single output file with UUID and
   source file tracking columns
5. Write FITS file (with OBSERVATIONS and TRAJECTORIES extensions) and/or CSV

Example
-------
>>> from kbmod.fits_export import export_results
>>> obs_table, traj_table = export_results(
...     directories=["/path/to/results"],
...     output_path="/path/to/combined_observations.fits",
...     output_csv="/path/to/combined_observations.csv",
...     uuids=["abc123"],
... )
"""

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import tqdm

from kbmod.results import Results

logger = logging.getLogger(__name__)


def export_results(
    directories,
    output_path=None,
    output_csv=None,
    uuids=None,
    glob_pattern="*.search.parquet",
    chunk_size=100000,
):
    """Export KBMOD result trajectories to combined FITS and/or CSV files.

    Scans one or more directories for result parquet files, processes
    them in memory-efficient chunks, and writes output files containing
    all observations with UUID and source file tracking.

    The FITS output contains two extensions:
    - OBSERVATIONS: One row per valid observation with RA/Dec/MJD
    - TRAJECTORIES: One row per trajectory with core parameters

    Parameters
    ----------
    directories : `str`, `Path`, or `list`
        One or more directory paths to scan, or direct paths to
        individual ``.search.parquet`` files.
    output_path : `str` or `Path`, optional
        Path for the output FITS file. If None, no FITS is written.
    output_csv : `str` or `Path`, optional
        Path for the output CSV file (observations only). If None, no CSV.
    uuids : `set` or `list` of `str`, optional
        If provided, only export these UUIDs. If ``None``, export all.
    glob_pattern : `str`
        Glob pattern for finding result files within directories.
        Default: ``"*.search.parquet"``.
    chunk_size : `int`
        Number of rows to read at a time. Default: ``100000``.

    Returns
    -------
    obs_table : `astropy.table.Table`
        The combined observations table (one row per valid observation).
        Columns: uuid, source_file, obs_index, ra_deg, dec_deg, mjd
    traj_table : `astropy.table.Table`
        The trajectory summary table (one row per trajectory).
        Columns: uuid, source_file, x, y, vx, vy, likelihood, flux, obs_count
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]

    # Normalize UUID set for fast lookup
    uuid_set = set(uuids) if uuids is not None else None

    # Accumulate observations and trajectories
    obs_tables = []
    traj_rows = []

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
        empty_obs = Table(
            {
                "uuid": np.array([], dtype=str),
                "source_file": np.array([], dtype=str),
                "obs_index": np.array([], dtype=np.int32),
                "ra_deg": np.array([], dtype=np.float64),
                "dec_deg": np.array([], dtype=np.float64),
                "mjd": np.array([], dtype=np.float64),
            }
        )
        empty_traj = Table(
            {
                "uuid": np.array([], dtype=str),
                "source_file": np.array([], dtype=str),
                "x": np.array([], dtype=np.int64),
                "y": np.array([], dtype=np.int64),
                "vx": np.array([], dtype=np.float64),
                "vy": np.array([], dtype=np.float64),
                "likelihood": np.array([], dtype=np.float64),
                "flux": np.array([], dtype=np.float64),
                "obs_count": np.array([], dtype=np.int64),
            }
        )
        return empty_obs, empty_traj

    logger.info(f"Found {len(result_paths)} result file(s) to process")

    for result_path in tqdm.tqdm(result_paths, desc="Result files"):
        logger.info(f"Processing: {result_path}")
        source_file_str = str(result_path)

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

                    # Add trajectory row
                    traj_rows.append(
                        {
                            "uuid": row_uuid,
                            "source_file": source_file_str,
                            "x": int(row["x"]),
                            "y": int(row["y"]),
                            "vx": float(row["vx"]),
                            "vy": float(row["vy"]),
                            "likelihood": float(row["likelihood"]),
                            "flux": float(row["flux"]),
                            "obs_count": int(row["obs_count"]),
                        }
                    )

                    # Add observation rows if any valid
                    if n_valid > 0:
                        img_ra = np.asarray(row["img_ra"])
                        img_dec = np.asarray(row["img_dec"])

                        obs_table = Table(
                            {
                                "uuid": [row_uuid] * n_valid,
                                "source_file": [source_file_str] * n_valid,
                                "obs_index": valid_idx.astype(np.int32),
                                "ra_deg": img_ra[valid_idx],
                                "dec_deg": img_dec[valid_idx],
                                "mjd": mjd_mid[valid_idx],
                            }
                        )
                        obs_tables.append(obs_table)

                        logger.debug(f"UUID {row_uuid}: extracted {n_valid} observations")

        except Exception as e:
            logger.error(f"Error processing {result_path}: {e}")
            raise

    # Combine all observation tables
    if obs_tables:
        combined_obs = vstack(obs_tables)
    else:
        combined_obs = Table(
            {
                "uuid": np.array([], dtype=str),
                "source_file": np.array([], dtype=str),
                "obs_index": np.array([], dtype=np.int32),
                "ra_deg": np.array([], dtype=np.float64),
                "dec_deg": np.array([], dtype=np.float64),
                "mjd": np.array([], dtype=np.float64),
            }
        )

    # Combine trajectory rows
    if traj_rows:
        traj_table = Table(rows=traj_rows)
    else:
        traj_table = Table(
            {
                "uuid": np.array([], dtype=str),
                "source_file": np.array([], dtype=str),
                "x": np.array([], dtype=np.int64),
                "y": np.array([], dtype=np.int64),
                "vx": np.array([], dtype=np.float64),
                "vy": np.array([], dtype=np.float64),
                "likelihood": np.array([], dtype=np.float64),
                "flux": np.array([], dtype=np.float64),
                "obs_count": np.array([], dtype=np.int64),
            }
        )

    logger.info(f"Total observations: {len(combined_obs)}")
    logger.info(f"Total trajectories: {len(traj_table)}")

    # Write FITS file
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["COMMENT"] = "KBMOD observations export"
        primary_hdu.header["N_OBS"] = len(combined_obs)
        primary_hdu.header["N_TRAJ"] = len(traj_table)

        obs_hdu = fits.BinTableHDU(combined_obs, name="OBSERVATIONS")
        obs_hdu.header["EXTNAME"] = "OBSERVATIONS"
        obs_hdu.header["COMMENT"] = "Per-observation RA/Dec/MJD data"

        traj_hdu = fits.BinTableHDU(traj_table, name="TRAJECTORIES")
        traj_hdu.header["EXTNAME"] = "TRAJECTORIES"
        traj_hdu.header["COMMENT"] = "Per-trajectory summary information"

        hdu_list = fits.HDUList([primary_hdu, obs_hdu, traj_hdu])
        hdu_list.writeto(str(output_path), overwrite=True)
        logger.info(f"Wrote FITS file: {output_path}")

    # Write CSV file (observations only, with trajectory info joined)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Join trajectory info to observations for a flat CSV
        # Create a lookup dict for trajectory info
        traj_lookup = {row["uuid"]: row for row in traj_table}

        csv_rows = []
        for obs_row in combined_obs:
            uuid = obs_row["uuid"]
            traj = traj_lookup.get(uuid, {})
            csv_rows.append(
                {
                    "uuid": uuid,
                    "source_file": obs_row["source_file"],
                    "obs_index": obs_row["obs_index"],
                    "ra_deg": obs_row["ra_deg"],
                    "dec_deg": obs_row["dec_deg"],
                    "mjd": obs_row["mjd"],
                    "x": traj.get("x", ""),
                    "y": traj.get("y", ""),
                    "vx": traj.get("vx", ""),
                    "vy": traj.get("vy", ""),
                    "likelihood": traj.get("likelihood", ""),
                    "flux": traj.get("flux", ""),
                    "obs_count": traj.get("obs_count", ""),
                }
            )

        if csv_rows:
            csv_table = Table(rows=csv_rows)
        else:
            csv_table = Table(
                {
                    "uuid": [],
                    "source_file": [],
                    "obs_index": [],
                    "ra_deg": [],
                    "dec_deg": [],
                    "mjd": [],
                    "x": [],
                    "y": [],
                    "vx": [],
                    "vy": [],
                    "likelihood": [],
                    "flux": [],
                    "obs_count": [],
                }
            )

        csv_table.write(str(output_csv), format="csv", overwrite=True)
        logger.info(f"Wrote CSV file: {output_csv}")

    return combined_obs, traj_table


def read_fits_export(fits_path):
    """Read back the tables from a FITS export file.

    Parameters
    ----------
    fits_path : `str` or `Path`
        Path to the FITS file created by ``export_results``.

    Returns
    -------
    obs_table : `astropy.table.Table`
        The observations table.
    traj_table : `astropy.table.Table`
        The trajectories table.
    """
    with fits.open(fits_path) as hdul:
        obs_table = Table.read(hdul["OBSERVATIONS"])
        traj_table = Table.read(hdul["TRAJECTORIES"])
    return obs_table, traj_table
