"""A program to ingest LSST Butler collections as KBMOD ImageCollections.

This script allows you to ingest collections from an LSST Butler repository into KBMOD's ImageCollection format.

To use this script, you need to have the LSST stack installed and set up properly and KBMOD available in your Python environment.
You however do not need GPU access within KBMOD to use this script.

To ingest a one or more collections from a Butler repository with a specific dataset type, you can run:
        `python kbmod_build_ic.py <repo_path> <datasetType> --collections <collection1> <collection2> ... --output_dir <output_directory> --overwrite`

Examples of datasetTypes include `calexp`, `preliminary_visit_image`, or `difference_image`.

If you want to ingest all Butler collections matching a regex pattern, you can run:
        `python kbmod_build_ic.py <repo_path> <datasetType> --collection_regex <regex_pattern> --output_dir <output_directory> --overwrite`

The ingested ImageCollections will be saved as `.ecsv` files in the specified output directory, with the original Butler collection name as the filename (substituting slashes with underscores).

For testing purposes, you can run with the `--dry` flag to see the sizes of the Butler collections without actually ingesting them, or
you can use the `--max_exposures` flag to limit the number of exposures processed per collection.
"""

import argparse
import logging
import os
import re

try:
    import lsst.daf.butler as dafButler
except ImportError:
    raise ImportError(
        "LSST stack not found. Please install the LSST stack to use this tool. (you may need to re-run `setup lsst_distrib`)"
    )

import kbmod

logger = kbmod.Logging.getLogger(__name__)


def ingest_collection(
    butler,
    collection_name,
    datasetType,
    target=None,
    max_exposures=None,
    output_dir=None,
    overwrite=False,
):
    """
    Ingest a single collection from the LSST Butler repository into a KBMOD ImageCollection, which is then saved as an ECSV file.

    Parameters
    ----------
    butler : dafButler.Butler
        The LSST Butler instance to use for querying datasets.
    collection_name : str
        The name of the collection to ingest.
    datasetType : str
        The dataset type to query from the collection (e.g., 'preliminary_visit_image', 'difference_image').
    target: str, optional
        The target name to use for the collection. If provided, will be used to filter collections.
    max_exposures : int, optional
        Maximum number of exposures to process from the collection. If None, all exposures are processed.
    output_dir : str, optional
        Directory to write the ImageCollection file. If None, no file is written.
    overwrite : bool, optional
        If True, overwrite existing ImageCollection files. Default is False.
    """
    if output_dir is not None:
        # Generate an output path based on the collection name and replace slashes with underscores
        output_collection_name = collection_name.replace("/", "_")
        output_path = os.path.join(output_dir, f"{output_collection_name}.ecsv")
        if not overwrite and os.path.exists(output_path):
            logger.info(f"Skipping {collection_name} as it already exists.")
            return
        logger.info(f"Preparing to use output path {output_path} for {collection_name}")

    # Get all butler references for the specified dataset type in this collection

    try:
        if target is None:
            refs = butler.registry.queryDatasets(datasetType, collections=[collection_name])
        else:
            refs = butler.query_datasets(
                datasetType,
                where=f"instrument='LSSTCam' and exposure.target_name='{target}'",
                collections=[collection_name],
            )
    except Exception as e:
        logger.error(f"Error querying collection {collection_name}: {e}")
        return
    refs = list(refs)
    if max_exposures is not None:
        refs = refs[: min(len(refs), max_exposures)]
        logger.info(f"Limiting to first {max_exposures} exposures for collection {collection_name}")
    if not refs:
        logger.debug(f"No datasets found for {datasetType} in {collection_name}.")
        return

    logger.debug(f"Creating ImageCollection for collection {collection_name}")
    ic = kbmod.ImageCollection.fromTargets(refs, butler=butler, force="ButlerStandardizer")
    logger.debug(f"Created ImageCollection for collection {collection_name} with {len(ic)} images")
    ic["collection"] = collection_name

    if output_dir is not None:
        if not overwrite and os.path.exists(output_path):
            # Check again if we should overwrite the output path due to potential parallel processing
            logger.debug(f"Output path {output_path} already exists, skipping write.")
            return
        ic.write(output_path, overwrite=overwrite)

    logger.info(f"Finished ingesting {collection_name}.")


def execute(args):
    # Instantiate the Butler
    butler = dafButler.Butler(args.repo)

    # Determine collections to ingest
    if args.collections:
        collections = args.collections
    else:
        # Get all collections that match the regex
        all_collections = butler.registry.queryCollections()
        collections = [col for col in all_collections if re.match(args.collection_regex, col)]

    if args.dry:
        # If a dry run, just print the sizes of the collections
        where_filter = "instrument='LSSTCam'"
        if args.target is not None:
            where_filter += f"and exposure.target_name='{args.target}'"

        all_count = len(
            list(
                butler.query_datasets(
                    args.datasetType,
                    where=where_filter,
                    collections=collections,
                )
            )
        )
        print(f"Found {len(collections)} collections matching the regex {args.collection_regex}.")
        print(f"Total exposures across all collections: {all_count}")
        return 0

    # Ingest each collection
    for collection_name in collections:
        logger.info(f"Ingesting {collection_name}")
        ingest_collection(
            butler,
            collection_name,
            args.datasetType,
            args.target,
            args.max_exposures,
            args.output_dir,
            args.overwrite,
        )


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod_build_ic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Ingest LSST Butler collections as KBMOD ImageCollections.",
    )
    # Required positional arguments
    parser.add_argument(
        "repo",
        help="Path to the LSST Butler repository",
    )
    parser.add_argument(
        "datasetType",
        help="Dataset type to ingest (e.g., 'preliminary_visit_image', 'difference_image')",
    )

    # Optional arguments for collection selection and ingestion. At least one of these must be provided.
    parser.add_argument(
        "--collections",
        nargs="+",
        help="Names of Butler collections to ingest separated by spaces. If provided, cannot also provide --collection_regex.",
    )
    parser.add_argument(
        "--collection_regex",
        help="Regex to match collection names",
    )
    parser.add_argument(
        "--target",
        help="Target name to use for the collection. If provided, will be used to filter collections for this target field.",
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        help="Directory to write ImageCollections",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ImageCollections",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Print collection sizes instead of ingesting",
    )
    parser.add_argument(
        "--max_exposures",
        type=int,
        help="Maximum number of exposures to process per collection",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Validate that at least one of collections or collection_regex is provided (but not both)
    if args.collections is None and args.collection_regex is None:
        parser.error("Must specify either --collections or --collection_regex for ingesting.")
    if args.collections is not None and args.collection_regex is not None:
        parser.error("Cannot specify both --collections and --collection_regex. Use one or the other.")

    execute(args)


if __name__ == "__main__":
    main()
