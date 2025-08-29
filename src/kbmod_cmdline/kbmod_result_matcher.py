"""A program to match one or more KBMOD results files against an ephemeris of known objects.

This script is designed to be run from the commandline and will process results files, matching them against known objects in a SkyTable database.
It will output a tbale of which results mathched with which known objects, parameters used, and any exceptions encountered during processing.

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

from astropy.table import Table

import csv

csv.field_size_limit(131072 * 2)  # Increase field size limit for reading large CSV files.

from kbmod import ImageCollection
from kbmod.filters.known_object_filters import KnownObjsMatcher
from kbmod.results import Results


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
        from astropy.coordinates import EarthLocation

        # Apply reflex correction to the RA and Dec columns.
        corrected_skycoord, _ = correct_parallax_geometrically_vectorized(
            ephem_table["RA"],
            ephem_table["Dec"],
            ephem_table["mjd_mid"],
            barycentric_distance=barycentric_dist,
            point_on_earth=EarthLocation.of_site(obs_site),
        )
        ephem_table[f"ra_{barycentric_dist}"] = corrected_skycoord.ra.deg
        ephem_table[f"dec_{barycentric_dist}"] = corrected_skycoord.dec.deg
    if barycentric_dist == 0.0:
        # If no correction is applied, just copy the original RA and Dec columns.
        ephem_table[f"ra_{barycentric_dist}"] = ephem_table["RA"]
        ephem_table[f"dec_{barycentric_dist}"] = ephem_table["Dec"]
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


def process_results_file(
    results_file,
    ephem_table,
    barycentric_dist,
    sep_thresh,
    time_thresh_s,
    min_obs,
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
    verbose : bool, optional
        If True, print verbose output. Default is False.
    max_results : int, optional
        Maximum number of results to process from the file. If None, all results will be processed

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the matched results with columns for results file, parameters used, uuid, and matched object name.
    """
    try:
        res = Results.read_table(results_file)
    except Exception as e:
        print(f"Error reading in results file {results_file}: {e}")
    if verbose:
        print(f"Processing {len(res)} results from file: {results_file}")
    if max_results is not None:
        if max_results <= 0:
            raise ValueError("max_results must be a positive integer.")
        if max_results < len(res):
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
    if args.results is None == args.results_glob is None:
        raise ValueError("You must provide either --results or --results_glob.")
    results_files = []
    if args.results is not None:
        if args.verbose:
            print(f"Using single results file: {args.results}")
        results_files.append(args.results)
    elif args.results_glob is not None:
        results_files = glob.glob(args.results_glob)
        if not results_files:
            raise ValueError(f"No files found matching glob pattern: {args.results_glob}")
    else:
        raise ValueError("You must provide either --results or --results_glob.")

    # Ensure the output directory exists.
    if args.output and not os.path.exists(args.output):
        if args.verbose:
            print(f"Creating output directory: {args.output}")
        os.makedirs(args.output)

    # Limit the number of files to process if max_files is set.
    if args.max_files is not None and args.max_files > len(results_files):
        results_files = results_files[: args.max_files]
        if args.verbose:
            print(f"Limiting to processing only {len(results_files)} results files.")

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

    # Reflex-correct ephems table if needed
    if (
        f"ra_{args.barycentric_dist}" in ephem_table.columns
        and f"dec_{args.barycentric_dist}" in ephem_table.columns
    ):
        print(
            f"Using existing columns 'ra_{args.barycentric_dist}' and 'dec_{args.barycentric_dist}' for matching."
        )
    else:
        ephem_table = reflex_correct_ephem_table(ephem_table, args.barycentric_dist, args.obs_site)
        # Write out the ephems table to the output directory if it doesn't already exist
        ephem_file = os.path.join(args.output, f"ephem_{args.barycentric_dist}.parquet")
        if not os.path.exists(ephem_file) or args.overwrite:
            ephem_table.write(ephem_file, format="parquet", overwrite=args.overwrite)
            if args.verbose:
                print(f"Saved reflex-corrected ephemeris table to: {ephem_file}")
        else:
            if args.verbose:
                print(f"Ephemeris file already exists: {ephem_file}. Skipping save.")

    matched_results_file = os.path.join(args.output, "matching_results.csv")
    exceptions_file = os.path.join(args.output, "exceptions.csv")
    if not args.overwrite:
        if os.path.exists(matched_results_file):
            raise ValueError(
                f"Matched results file already exists: {matched_results_file}. Use --overwrite to overwrite."
            )
        if os.path.exists(exceptions_file):
            raise ValueError(
                f"Exceptions file already exists: {exceptions_file}. Use --overwrite to overwrite."
            )

    # Process each result file, tracking any exceptions that occur during processing.
    exceptions = {"result_file": [], "error": []}
    first_write = True
    for i, results_file in enumerate(results_files):
        if args.verbose:
            print(f"Processing results file {i+1}/{len(results_files)}: {results_file}")
        try:
            results_processed = process_results_file(
                results_file,
                ephem_table,
                args.barycentric_dist,
                args.sep_thresh,
                args.time_thresh_s,
                args.min_obs,
                max_results=args.max_results,
            )

            if first_write:
                if os.path.exists(matched_results_file):
                    if not args.overwrite:
                        raise ValueError(
                            f"Manifest file already exists: {matched_results_file}. Use --overwrite to overwrite."
                        )
                    if args.verbose:
                        print(f"Overwriting existing matching output file: {matched_results_file}")
                    os.remove(matched_results_file)
                if os.path.exists(exceptions_file):
                    if not args.overwrite:
                        raise ValueError(
                            f"Exceptions file already exists: {exceptions_file}. Use --overwrite to overwrite."
                        )
                    if args.verbose:
                        print(f"Overwriting existing exceptions output file: {exceptions_file}")
                    os.remove(exceptions_file)
                # The outputted file is a serialized CSV file of a pandas DataFrame.
                results_processed.to_csv(matched_results_file, mode="a", header=True, index=False)
                first_write = False
            else:
                # The outputted file file is a serialized CSV file of a pandas DataFrame.
                # We output in append mode if it already exists.
                results_processed.to_csv(matched_results_file, mode="a", header=False, index=False)
        except Exception as e:
            print(f"Exception occurred: {e}")
            exceptions["result_file"].append(results_file)
            exceptions["error"].append(str(e))

    if len(exceptions) > 0:
        print("Exceptions occurred during processing. Writing out exceptions.")
        pd.DataFrame(exceptions).to_csv(exceptions_file, index=False)
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
        "--max_results",
        type=int,
        default=None,
        help="Maximum number of results to process from each file. If None, all results will be processed.",
    )

    args = parser.parse_args()
    execute(args)
