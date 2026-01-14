"""A program to match one or more KBMOD results files against an ephemeris of known objects.

This script is designed to be run from the commandline and will process results files, matching them against known objects.
It will output a table of which results matched with which known objects, parameters used, and any exceptions encountered during processing.

Usage:
    python kbmod_result_matcher.py --results_glob <results_file> --ephem <ephemeris_file> [--output <output_dir>]

Example:
    python kbmod_result_matcher.py --results /output/resuls*.parquet --ephem skytable.parquet --output .

This will produce a CSV file, `matching_results.csv` in the output directory with the matched results, and an `exceptions.csv` file with any files that could not be processed.
"""

import argparse
import glob
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from astropy.table import Table

import csv

csv.field_size_limit(131072 * 2)  # Increase field size limit for reading large CSV files.

from kbmod import ImageCollection
from kbmod.filters.known_object_filters import KnownObjsMatcher
from kbmod.results import Results

from astropy.coordinates import Angle, EarthLocation
from astropy.time import Time
import astropy.units as u
import numpy as np
import pyarrow.parquet as pq

from kbmod.wcs_utils import deserialize_wcs


def clean_ephem_table(ephem):
    """
    Standardize ephemeris table columns and formats.
    Handles both JPL and Skybot ephemeris formats.
    """

    def clean_dec_string(dec_str):
        # Replace delimiters to get '-17 23 27.0' which astropy can better parse
        return dec_str.replace("'", " ").replace('"', "")

    # Convert RA if needed
    if "ra" not in ephem.colnames:
        if "RA" in ephem.colnames:
            ephem.rename_column("RA", "ra")
        elif "Astrometric RA (hh:mm:ss)" in ephem.colnames:
            # JPL format
            ephem["ra"] = Angle(ephem["Astrometric RA (hh:mm:ss)"], unit="hourangle").deg
        elif "RA (hms)" in ephem.colnames:
            # Skybot format
            ephem["ra"] = Angle(ephem["RA (hms)"], unit="hourangle").deg
        else:
            raise ValueError(
                f"Ephemeris table must contain 'ra' column for reflex correction. "
                f"Available columns: {ephem.colnames}"
            )

    if "dec" not in ephem.colnames:
        if "Dec" in ephem.colnames:
            ephem.rename_column("Dec", "dec")
        elif "Astrometric Dec (dd mm'ss\")" in ephem.colnames:
            # JPL format
            cleaned_decs = [clean_dec_string(s) for s in ephem["Astrometric Dec (dd mm'ss\")"]]
            ephem["dec"] = Angle(cleaned_decs, unit="deg").deg
        elif "DEC (dms)" in ephem.colnames:
            # Skybot format - already in degrees:minutes:seconds
            cleaned_decs = [clean_dec_string(s) for s in ephem["DEC (dms)"]]
            ephem["dec"] = Angle(cleaned_decs, unit="deg").deg
        else:
            raise ValueError(
                f"Ephemeris table must contain 'dec' column for reflex correction. "
                f"Available columns: {ephem.colnames}"
            )

    if "mjd_mid" not in ephem.colnames:
        if "obs-time" in ephem.colnames:
            # JPL format
            ephem["mjd_mid"] = [Time(t, scale="utc").mjd for t in ephem["obs-time"]]
        elif "ref_epoch" in ephem.colnames:
            # Skybot format - ref_epoch is already MJD or datetime string
            try:
                # Try as MJD float first
                ephem["mjd_mid"] = [float(t) for t in ephem["ref_epoch"]]
            except (ValueError, TypeError):
                # Fall back to parsing as datetime
                ephem["mjd_mid"] = [Time(t, scale="utc").mjd for t in ephem["ref_epoch"]]
        else:
            raise ValueError(
                f"Ephemeris table must contain 'mjd_mid' column for reflex correction. "
                f"Available columns: {ephem.colnames}"
            )

    if "Name" not in ephem.colnames:
        if "Clean Name" in ephem.colnames:
            ephem.rename_column("Clean Name", "Name")
        elif "Object name" in ephem.colnames:  # Common alias
            ephem.rename_column("Object name", "Name")
        else:
            raise ValueError(
                f"Ephemeris table must contain 'Name' column for matching. "
                f"Available columns: {ephem.colnames}"
            )
    return ephem


def reflex_correct_ephem_table(ephem_table, barycentric_dist, obs_site="Rubin"):
    """Apply reflex correction to the ephemeris table if barycentric distance is provided.

    Assumes that the observatory is Rubin Observatory.

    Produces columns of the form 'ra_<barycentric_dist>' and 'dec_<barycentric_dist>' in the returned table.

    Parameters
    ----------
    ephem_table : astropy.table.Table
        The ephemeris table containing known objects, assumes 'RA', 'Dec', and 'mjd_mid' columns.
    barycentric_dist : float
        The barycentric distance in AU. If 0.0, no correction is applied.
    obs_site : str
        The observatory site to use for reflex correction. Default is "Rubin".

    Returns
    -------
    astropy.table.Table
        The corrected ephemeris table.
    """

    if barycentric_dist != 0.0:
        from kbmod.reprojection_utils import correct_parallax_geometrically_vectorized

        # Apply reflex correction to the RA and Dec columns.
        corrected_skycoord, _ = correct_parallax_geometrically_vectorized(
            ephem_table["ra"],
            ephem_table["dec"],
            ephem_table["mjd_mid"],
            barycentric_distance=barycentric_dist,
            point_on_earth=EarthLocation.of_site(obs_site),
        )
        ephem_table[f"ra_{barycentric_dist}"] = corrected_skycoord.ra.deg
        ephem_table[f"dec_{barycentric_dist}"] = corrected_skycoord.dec.deg
    else:
        # If no correction is applied (distance of 0.0), just copy the original RA and Dec columns.
        ephem_table[f"ra_{barycentric_dist}"] = ephem_table["ra"]
        ephem_table[f"dec_{barycentric_dist}"] = ephem_table["dec"]
    return ephem_table


def get_ic_from_results_file(res_filepath):
    """Get the ImageCollection from the results filepath.

    This works for results files outputted from our KBMOD parsl workflow because
    1. We copy the ImageCollection to the same directory as the results file.
    2. The workflow outputs files with appended file extensions from the paths of previous steps.
       e.g. `patch_33.collection.[other_steps].results.parqute` for a results file and `patch_33.collection` for the ImageCollection.

    Parameters
    ----------
    res_filepath : str
        The path to the results file.

    Returns
    -------
    str
        The path to the corresponding ImageCollection file.
    """
    # An ImageCollection will be in the same directory as the results file but will not have the extra extensions
    # from subsquent KBMOD workflow steps.
    collection_idx = res_filepath.find(".collection")
    if collection_idx == -1:
        raise ValueError(f"Could not find .collection in {res_filepath} path")
    ic_path = res_filepath[:collection_idx] + ".collection"
    if not os.path.exists(ic_path):
        raise ValueError(f"ImageCollection file not found: {ic_path} for results file {res_filepath}")
    return ic_path


def _process_results_file_chunks(
    results_file,
    ephem_table,
    barycentric_dist,
    sep_thresh,
    time_thresh_s,
    min_obs,
    chunk_size,
    verbose=False,
    max_results=None,
):
    """Internal helper to process a results file in chunks using Results.read_table_chunks."""

    # We defer initializing the matcher until we have the first chunk,
    # because we want to prioritize the mjd_mid from the Results file itself
    # (which ensures consistency with the obs_valid array).
    known_objs_matcher = None

    all_matching_dfs = []
    total_processed = 0
    wcs_warned = False
    mjd_warned = False

    global_wcs = None
    try:
        ic = ImageCollection.read(get_ic_from_results_file(results_file))
        from astropy.wcs import WCS

        global_wcs = WCS(ic.data[0]["global_wcs"])
    except Exception as e:
        if verbose:
            print(f"  Failed to recover WCS from ImageCollection: {e}")
        raise e

    # Use the generator from Results class
    for res in Results.read_table_chunks(results_file, chunk_size):
        # Initialize matcher on first chunk
        if known_objs_matcher is None:
            if verbose:
                print(f"  Using mjd_mid from Results metadata: {len(res.mjd_mid)} observations")

            # Create the matcher once (reused across all chunks)
            known_objs_matcher = KnownObjsMatcher(
                ephem_table,
                res.mjd_mid,
                matcher_name="known_matcher",
                sep_thresh=sep_thresh,
                time_thresh_s=time_thresh_s,
                name_col="Name",
                ra_col=f"ra_{barycentric_dist}",
                dec_col=f"dec_{barycentric_dist}",
                mjd_col="mjd_mid",
            )

        if verbose:
            print(f"  Processing batch {mjd_warned + 1} (size {len(res)})...")
        mjd_warned += 1  # Utilizing this unused variable as batch counter since I can't change init easily

        if max_results is not None and total_processed >= max_results:
            break

        # Limit the chunk if we only need a few more results
        if max_results is not None:
            remaining = max_results - total_processed
            if len(res) > remaining:
                res.table = res.table[:remaining]

        # Extract WCS from the chunk if available, or try IC again if missing
        if res.wcs is None:
            if not wcs_warned:
                if verbose:
                    print("  WCS missing in chunk, using ImageCollection global_wcs...")
                wcs_warned = True  # Don't spam warnings
            res.wcs = global_wcs  # Always apply fallback WCS to each chunk

        # Check if WCS is still None - we cannot proceed without it
        if res.wcs is None:
            raise ValueError(
                f"No WCS available for results file. Cannot perform sky coordinate matching. "
                f"Ensure the result file or its ImageCollection contains WCS information."
            )

        # Run matching
        # Note: res.mjd_mid might be set from metadata, but we use the one we loaded for the matcher
        # The matcher uses its own self.obstimes, but it needs 'obs_valid' from res to match dimensions.

        try:
            known_objs_matcher.match(res, res.wcs)
            known_objs_matcher.match_on_min_obs(res, min_obs)
        except Exception as e:
            print(f"  Match error in chunk: {e}")
            if verbose:
                print(f"  Chunk size: {len(res)}")
                if "obs_valid" in res.colnames:
                    print(f"  obs_valid shape: {res['obs_valid'].shape}")
            raise e

        # Extract matches
        uuids = []
        matched_names = []
        matched_obs = []
        obs_ratios = []

        for row in res:
            if row["known_matcher"] is not None:
                for match, obs_mask in row["known_matcher"].items():
                    uuids.append(row["uuid"])
                    matched_names.append(match)
                    num_matched_obs = sum(obs_mask)
                    matched_obs.append(num_matched_obs)
                    obs_ratios.append(num_matched_obs / len(obs_mask))

        if uuids:
            chunk_df = pd.DataFrame(
                {
                    "results_file": [results_file] * len(uuids),
                    "barycentric_dist": [barycentric_dist] * len(uuids),
                    "sep_thresh": [sep_thresh] * len(uuids),
                    "time_thresh_s": [time_thresh_s] * len(uuids),
                    "min_obs": [min_obs] * len(uuids),
                    "uuid": uuids,
                    "name": matched_names,
                    "matched_obs": matched_obs,
                    "obs_ratio": obs_ratios,
                }
            )
            all_matching_dfs.append(chunk_df)

        total_processed += len(res)
        if verbose and total_processed % (chunk_size * 10) == 0:
            print(f"  Processed {total_processed} rows...")

    if all_matching_dfs:
        return pd.concat(all_matching_dfs, ignore_index=True)
    else:
        return pd.DataFrame(
            {
                "results_file": [],
                "barycentric_dist": [],
                "sep_thresh": [],
                "time_thresh_s": [],
                "min_obs": [],
                "uuid": [],
                "name": [],
                "matched_obs": [],
                "obs_ratio": [],
            }
        )


def _process_file_worker(
    results_file,
    ephem_table,
    barycentric_dist,
    sep_thresh,
    time_thresh_s,
    min_obs,
    chunk_size,
    verbose,
    max_results,
):
    """Worker function for multiprocessing.

    Returns a tuple of (results_file, result_df, error) where error is None on success.
    """
    try:
        result_df = process_results_file(
            results_file,
            ephem_table,
            barycentric_dist,
            sep_thresh,
            time_thresh_s,
            min_obs,
            chunk_size=chunk_size,
            verbose=verbose,
            max_results=max_results,
        )
        return (results_file, result_df, None)
    except Exception as e:
        return (results_file, None, str(e))


def process_results_file(
    results_file,
    ephem_table,
    barycentric_dist,
    sep_thresh,
    time_thresh_s,
    min_obs,
    chunk_size=None,
    verbose=False,
    max_results=None,
):
    """Process a single results file against the ephemeris of known objects.

    Parameters
    ----------
    results_file : str
        The path to the results file to process.
    ephem_table : pandas.DataFrame
        The ephemeris table containing known objects. Assumes `ra_<barycentric_dist>`, `dec_<barycentric_dist>`, and `mjd_mid` columns.
    barycentric_dist : float
        The barycentric distance in AU used for reflex correction.
    sep_thresh : float
        The separation threshold in arcseconds for matching results to known objects.
    time_thresh_s : float
        The time threshold in seconds for matching results to known objects.
    min_obs : int
        Minimum number of observations required to consider a match valid.
    chunk_size : int, optional
        Process file in chunks of this size. If None, process whole file.
    verbose : bool, optional
        If True, print verbose output. Default is False.
    max_results : int, optional
        Maximum number of results to process from the file. If None, all results will be processed

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the matched results with columns for results file, parameters used, uuid, and matched object name.
    """

    if verbose:
        print(f"Processing results from file: {results_file}")

    # Use chunked processing if chunk_size is specified
    if chunk_size is not None:
        return _process_results_file_chunks(
            results_file,
            ephem_table,
            barycentric_dist,
            sep_thresh,
            time_thresh_s,
            min_obs,
            chunk_size,
            verbose,
            max_results,
        )

    try:
        res = Results.read_table(results_file)
    except Exception as e:
        print(f"Error reading in results file {results_file}: {e}")
    if verbose:
        print(f"Processing {len(res)} results from file: {results_file}")
    if max_results is not None:
        if max_results <= 0:
            raise ValueError("max_results must be a positive integer.")
        if verbose and max_results < len(res):
            print(f"Limiting processing to the first {max_results} results.")
            res.table = res.table[:max_results]
    known_objs_matcher = KnownObjsMatcher(
        ephem_table,
        res.mjd_mid,
        matcher_name="known_matcher",
        sep_thresh=sep_thresh,
        time_thresh_s=time_thresh_s,
        name_col="Name",
        ra_col=f"ra_{barycentric_dist}",
        dec_col=f"dec_{barycentric_dist}",
        mjd_col="mjd_mid",
    )
    if verbose:
        print(f"Finished matching results in {results_file} against known objects.")

    # Get the global WCS from the results for matching.
    wcs = res.wcs
    if wcs is None:
        # Get the global_wcs string from the first row of the ImageCollection as a WCS object.
        ic = ImageCollection.read(get_ic_from_results_file(results_file))
        from astropy.wcs import WCS

        wcs = WCS(ic[0]["global_wcs"])

    # Carry out initial matching to known objects and populate the matches column.
    known_objs_matcher.match(res, wcs)

    # Filter the matches down to results with at least min_obs observations.
    known_objs_matcher.match_on_min_obs(res, min_obs)

    uuids = []
    matched_names = []
    # The number of matched observations for each result and object.
    matched_obs = []
    # The propotion of observations for a result that were matched to a known object.
    obs_ratios = []
    for row in res:
        if row["known_matcher"] is not None:
            for match, obs_mask in row["known_matcher"].items():
                uuids.append(row["uuid"])
                matched_names.append(match)
                # Sum the boolean mask to get the number of matched observations for this result and object.
                num_matched_obs = sum(obs_mask)
                matched_obs.append(num_matched_obs)
                obs_ratios.append(num_matched_obs / len(obs_mask))
    matching_df = pd.DataFrame(
        {
            "results_file": [results_file] * len(uuids),
            "barycentric_dist": [barycentric_dist] * len(uuids),
            "sep_thresh": [sep_thresh] * len(uuids),
            "time_thresh_s": [time_thresh_s] * len(uuids),
            "min_obs": [min_obs] * len(uuids),
            "uuid": uuids,
            "name": matched_names,
            "matched_obs": matched_obs,
            "obs_ratio": obs_ratios,
        }
    )
    if verbose:
        print(f"Found {len(matching_df)} matches in {results_file}.")
    return matching_df


def execute(args):
    # Confirm the ephems esists before potentially globbing files.
    if args.ephem is None:
        raise ValueError("You must provide the path to the ephemeris file using --ephem.")
    if not os.path.exists(args.ephem):
        raise FileNotFoundError(f"Ephemeris file not found: {args.ephem}")

    # Generate the list of results files to process.
    if args.results is None == args.results_glob is None == args.results_file_list is None:
        raise ValueError("You must provide either --results, --results_glob, or --results_file_list.")
    results_files = []
    if args.results is not None:
        if args.verbose:
            print(f"Using single results file: {args.results}")
        results_files.append(args.results)
    elif args.results_file_list is not None:
        if args.verbose:
            print(f"Reading results files from: {args.results_file_list}")
        with open(args.results_file_list, "r") as f:
            results_files = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if not results_files:
            raise ValueError(f"No files found in results file list: {args.results_file_list}")
        if args.verbose:
            print(f"Found {len(results_files)} results files in list.")
    elif args.results_glob is not None:
        results_files = glob.glob(args.results_glob, recursive=True)
        if not results_files:
            raise ValueError(f"No files found matching glob pattern: {args.results_glob}")
    else:
        raise ValueError("You must provide either --results, --results_glob, or --results_file_list.")

    # Convert all result files to absolute paths for consistency
    results_files = [os.path.abspath(f) for f in results_files]

    # Ensure the output directory exists.
    if args.output and not os.path.exists(args.output):
        if args.verbose:
            print(f"Creating output directory: {args.output}")
        os.makedirs(args.output)

    # Limit the number of files to process if max_files is set.
    if args.max_files is not None and args.max_files < len(results_files):
        results_files = results_files[: args.max_files]
        if args.verbose:
            print(f"Limiting to processing only {len(results_files)} results files.")

    # Filter out files that are too large.
    if args.max_file_size_gb is not None:
        max_bytes = args.max_file_size_gb * 1024 * 1024 * 1024
        original_count = len(results_files)
        results_files = [f for f in results_files if os.path.getsize(f) <= max_bytes]
        skipped = original_count - len(results_files)
        if skipped > 0 and args.verbose:
            print(f"Skipped {skipped} files larger than {args.max_file_size_gb} GB.")

    # Open the ephemeris file as an astropoy Table
    if args.verbose:
        print(f"Loading ephemeris from: {args.ephem}")
    if args.ephem.endswith(".ecsv"):
        ephem_table = Table.read(args.ephem, format="ascii.ecsv")
    elif args.ephem.endswith(".csv"):
        ephem_table = Table.read(args.ephem)
    elif args.ephem.endswith(".parquet"):
        # load as a pandas dataframe to better handle type inference.
        ephem_table = Table.from_pandas(pd.read_parquet(args.ephem))
    else:
        raise ValueError(f"Unsupported ephemeris file format: {args.ephem}. Use .csv or .ecsv or .parquet")

    # Clean the ephemeris table first
    if args.verbose:
        print("Cleaning ephemeris table...")
    ephem_table = clean_ephem_table(ephem_table)

    # Reflex-correct ephems table if needed
    if (
        f"ra_{args.barycentric_dist}" in ephem_table.columns
        and f"dec_{args.barycentric_dist}" in ephem_table.columns
    ):
        print(
            f"Using existing columns 'ra_{args.barycentric_dist}' and 'dec_{args.barycentric_dist}' for matching."
        )
    else:
        if args.verbose:
            print(f"Applying reflex correction with dist={args.barycentric_dist}...")
        ephem_table = reflex_correct_ephem_table(ephem_table, args.barycentric_dist, args.obs_site)
        # Write out the ephems table to the output directory if it doesn't already exist
        ephem_file = os.path.join(args.output, f"ephem_{args.barycentric_dist}.parquet")
        if not os.path.exists(ephem_file) or args.overwrite:
            # We convert to pandas to write to parquet because astropy.table.write(format='parquet')
            # can be finicky with mixed object types or specific column names.
            ephem_df = ephem_table.to_pandas()
            ephem_df.to_parquet(ephem_file)
            if args.verbose:
                print(f"Saved reflex-corrected ephemeris table to: {ephem_file}")
        else:
            if args.verbose:
                print(f"Ephemeris file already exists: {ephem_file}. Skipping save.")

    matched_results_file = os.path.join(args.output, "matching_results.csv")
    exceptions_file = os.path.join(args.output, "exceptions.csv")
    processed_files = set()
    manifest_file = os.path.join(args.output, "manifest.txt")

    if args.overwrite:
        # Overwrite mode: clear everything
        for f in [manifest_file, matched_results_file, exceptions_file]:
            if os.path.exists(f):
                if args.verbose:
                    print(f"Removing existing file: {f}")
                os.remove(f)
    elif os.path.exists(manifest_file):
        # Resume mode
        if args.verbose:
            print(f"Loading processed files from manifest: {manifest_file}")
        with open(manifest_file, "r") as f:
            processed_files = set(os.path.abspath(line.strip()) for line in f if line.strip())
        if args.verbose:
            print(f"Resuming: {len(processed_files)} files already processed, will be skipped.")

    else:
        # Standard safety check if not resuming (and not overwriting)
        if os.path.exists(matched_results_file):
            raise ValueError(
                f"Matched results file already exists: {matched_results_file}. Use --overwrite to overwrite."
            )
        elif os.path.exists(exceptions_file):
            raise ValueError(
                f"Exceptions file already exists: {exceptions_file}. Use --overwrite to overwrite."
            )

    # Filter out already processed files
    files_to_process = [f for f in results_files if f not in processed_files]
    skipped_count = len(results_files) - len(files_to_process)
    if skipped_count > 0 and args.verbose:
        print(f"Skipping {skipped_count} already processed files from manifest.")

    # Process each result file, tracking any exceptions that occur during processing.
    exceptions = {"result_file": [], "error": []}

    if args.n_workers > 1 and len(files_to_process) > 1:
        # Parallel processing with multiprocessing Pool
        if args.verbose:
            print(f"Using {args.n_workers} parallel workers for {len(files_to_process)} files.")

        # Create worker function with fixed arguments using partial
        worker_fn = partial(
            _process_file_worker,
            ephem_table=ephem_table,
            barycentric_dist=args.barycentric_dist,
            sep_thresh=args.sep_thresh,
            time_thresh_s=args.time_thresh_s,
            min_obs=args.min_obs,
            chunk_size=args.chunk_size,
            verbose=False,  # Disable verbose in workers to avoid garbled output
            max_results=args.max_results,
        )

        with Pool(processes=args.n_workers) as pool:
            # Use imap_unordered for better progress tracking
            results_iter = pool.imap_unordered(worker_fn, files_to_process)
            for results_file, result_df, error in tqdm(
                results_iter, total=len(files_to_process), desc="Processing files", disable=not args.verbose
            ):
                if error is not None:
                    print(f"Exception in {results_file}: {error}")
                    exceptions["result_file"].append(results_file)
                    exceptions["error"].append(error)
                    # Stream exception to file
                    try:
                        header = not os.path.exists(exceptions_file)
                        ex_df = pd.DataFrame({"result_file": [results_file], "error": [error]})
                        ex_df.to_csv(exceptions_file, mode="a", header=header, index=False)
                    except Exception as write_err:
                        print(f"Failed to write exception: {write_err}")
                else:
                    # Write results sequentially (no contention)
                    header = not os.path.exists(matched_results_file)
                    result_df.to_csv(matched_results_file, mode="a", header=header, index=False)
                    with open(manifest_file, "a") as f:
                        f.write(f"{results_file}\n")
    else:
        # Sequential processing (original behavior)
        for results_file in tqdm(files_to_process, desc="Processing results files", disable=not args.verbose):
            try:
                results_processed = process_results_file(
                    results_file,
                    ephem_table,
                    args.barycentric_dist,
                    args.sep_thresh,
                    args.time_thresh_s,
                    args.min_obs,
                    chunk_size=args.chunk_size,
                    verbose=args.verbose,
                    max_results=args.max_results,
                )

                # Append results to output file
                header = not os.path.exists(matched_results_file)
                results_processed.to_csv(matched_results_file, mode="a", header=header, index=False)

                # Update manifest
                with open(manifest_file, "a") as f:
                    f.write(f"{results_file}\n")

            except Exception as e:
                print(f"Exception occurred: {e}")

                # Stream exception to file
                try:
                    header = not os.path.exists(exceptions_file)
                    ex_df = pd.DataFrame({"result_file": [results_file], "error": [str(e)]})
                    ex_df.to_csv(exceptions_file, mode="a", header=header, index=False)
                except Exception as write_err:
                    print(
                        f"Failed to write exception to file: {exceptions_file}. \n Error: {write_err}. \n Original exception: {e}"
                    )

                # Keep in memory for summary
                exceptions["result_file"].append(results_file)
                exceptions["error"].append(str(e))

    if len(exceptions["result_file"]) > 0:
        print(
            f"{len(exceptions['result_file'])} Exceptions occurred during processing. See {exceptions_file}"
        )
        if args.verbose:
            print(f"Some files could not be processed. See exceptions file: {exceptions_file}")

    if args.verbose:
        print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="kbmod-result-matcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A program to match KBMOD results files against known objects in a SkyTable database.",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="The file path for the input KBMOD Results file to process.",
    )
    parser.add_argument(
        "--results_glob",
        type=str,
        default=None,
        help="A glob pattern to match multiple results files. If provided, overrides --results.",
    )
    parser.add_argument(
        "--results_file_list",
        type=str,
        default=None,
        help="Path to a text file containing one results file path per line.",
    )

    parser.add_argument(
        "--ephem",
        type=str,
        required=True,
        help="The file path for the ephemeris of known objects.",
    )

    # Optional arguments below this line.

    # Arguments for output files.
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Directory to store the output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, overwrite existing output files.",
    )

    # Arguments for optional reflex-correction of ephemeris table.
    parser.add_argument(
        "--barycentric_dist",
        type=float,
        default=0.0,  # Default to no reflex correction.
        help="Assumed barycentric distance in au for reflex-correciton. 0.0 if no reflex-correction was applied.",
    )
    parser.add_argument(
        "--obs_site",
        type=str,
        default="Rubin",
        help="The observatory site to use for reflex correction. Default is Rubin Observatory.",
    )

    # Arguments for matching parameters.
    parser.add_argument(
        "--sep_thresh",
        type=float,
        default=5.0,
        help="Separation threshold in arcseconds for matching results to known objects.",
    )
    parser.add_argument(
        "--time_thresh_s",
        type=float,
        default=30.0,
        help="Time threshold in seconds for matching results to known objects.",
    )
    parser.add_argument(
        "--min_obs",
        type=int,
        default=1,
        help="Minimum number of observations required to consider a match valid.",
    )

    # Additional optional arugments for debugging and testing.
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process. If None, all files will be processed.",
    )
    parser.add_argument(
        "--max_file_size_gb",
        type=float,
        default=None,
        help="Skip result files larger than this size (in GB). Useful to avoid OOM.",
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=None,
        help="Maximum number of results to process from each file. If None, all results will be processed.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Process result files in chunks of this size (rows). Enables memory-efficient processing of large files.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing files. Default is 1 (sequential processing).",
    )

    args = parser.parse_args()
    execute(args)
