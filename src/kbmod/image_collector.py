try:
    import lsst.daf.butler as dafButler
except ImportError:
    raise ImportError(
        "LSST stack not found. Please install the LSST stack to use this module."
    )

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.table import Table

from urllib.parse import urlparse

import os
import re

import kbmod

logger = kbmod.Logging.getLogger(__name__)


def _chunked_data_ids(dataIds, chunk_size=200):
    """Helper function to yield successive chunk_size chunks from dataIds."""
    for i in range(0, len(dataIds), chunk_size):
        yield dataIds[i : i + chunk_size]


def _trim_uri(uri):
    """Trim the URI to a more standardized output."""
    return urlparse(uri).path


def _fake_ingest_collection(
    repo,
    collection_name,
    datasetType,
    output_dir=None,
    naming_scheme=None,
    overwrite=False,
):
    # Delete me
    butler = dafButler.Butler(repo)
    result = kbmod.ImageCollection.fromTargets(
        butler.registry.queryDatasets(datasetType, collections=[collection_name])
    )
    raise ValueError(f"Result is {result}")
    return result


def _ingest_collection(
    repo,
    collection_name,
    datasetType,
    output_dir=None,
    naming_scheme=None,
    overwrite=False,
):
    """
    Ingest the data from a butler collection into a saved ImageCollection.

    The ImageCollection is serialized as an ECSV file in the output directory
    with the name of the butler collection.

    Parameters
    ----------
    repo : `str`
        The path to the LSST butler repository.
    collection_name : `str`
        The name of the collection to ingest.
    datasetType : `str`
        The dataset type to ingest.
    output_dir : `str`
        The directory to write the ImageCollections to. If None, images are not written to disk.
    naming_scheme: `callable`, optional
        A function to use to name the output ImageCollection. Default is None.
    overwrite : `bool`, optional
        If True, overwrite existing ImageCollections. Default is False.

    Returns
    -------
    ic: `kbmod.ImageCollection`
        The ImageCollection created from the butler collection.
    """
    # Note that we replace the Butler collection name's with '/' to simplify directory structure
    if output_dir is not None:
        output_collection_name = ""
        if naming_scheme is None:
            # Butler collections frequently have '/' in their names, which can cause issues with file paths
            collection_name.replace("/", "_")
        else:
            # Use a user-specified naming schheme
            naming_scheme(collection_name)
        output_path = os.path.join(output_dir, f"{output_collection_name}.collection")
        if not overwrite and os.path.exists(output_path):
            logger.debug(f"Skipping {collection_name} as it already exists.")
            return
        # logger.debug(f"Preparing to use output path {output_path} for {collection_name}")

    # Initialize a butler for ingesting our collection
    butler = dafButler.Butler(repo)
    # logger.debug(f"Using butler {repo}")

    # Query the butler for the refs associated with the given dataset type and collection
    refs = butler.registry.queryDatasets(datasetType, collections=[collection_name])
    # logger.debug(f"Found {len(list(refs))} refs for {datasetType} in {collection_name}")
    if not refs:
        # logger.debug(f"No datasets found for {datasetType} in {collection_name}.")
        return

    # Create an ImageCollection for all matching refs. This should fetch assemble the metadata
    # used for RegionSearch and optionally later materialize a KBMOD work unit for further processing.
    # logger.debug("Creating ImageCollection")
    ic = kbmod.ImageCollection.fromTargets(
        refs, butler=butler, force="ButlerStandardizer"
    )

    # Add the original collection name to the ImageCollection
    # logger.debug(f"Created ImageCollection for collection {collection_name} with {len(ic)} images")
    ic["collection"] = collection_name

    # Check again if we should overwrite the output path due to parallel processing
    if output_dir is not None:
        if not overwrite and os.path.exists(output_path):
            logger.debug(
                f"Output path {output_path} was created while processing {collection_name}. Aborting."
            )
            return
        ic.write(output_path, overwrite=overwrite)

        # logger.debug("Creating bboxes")
        bbox_cols = [
            "ra",
            "dec",
            "ra_tl",
            "dec_tl",
            "ra_tr",
            "dec_tr",
            "ra_bl",
            "dec_bl",
            "ra_br",
            "dec_br",
            "mjd_mid",
            "dataId",
        ]
        bbox_path = os.path.join(output_dir, f"{collection_name}.bbox")
        ic[bbox_cols].write(
            bbox_path,
            format="ascii.ecsv",
            overwrite=True,  # Always overwrite if we wrote new ImageCollection
        )

    # logger.debug(f"Finished ingesting {collection_name}.")
    return ic


class ImageCollector:
    """
    A class for making use of the ButlerStandardizer to ingest one or more Butler
    collections as ImageCollections.

    Note that this is an intermediate step for the RegionSearch class, here we
    will assemble an intermediate representation it searches across in the form
    of an ImageCollection.
    """

    def __init__(self, repo, debug=False):
        """
        Parameters
        ----------
        repo : `str`
            The path to the LSST butler repository.
        """
        self.repo = repo
        self.debug = debug

    def get_collection_sizes(
        self, datasetType, collection_names=None, collection_regex=None
    ):
        """
        Returns the sizes of a group of collections for a given dataset type.

        Parameters
        ----------
        datasetType : `str`
            The dataset type to search for.
        collection_names : `list[str]`, optional
            The names of the collections to get the dataset sizes for. If None, use collection_regex.
        collection_regex : `str`, optional
            A regex pattern to match collection names. If None, use collection_names.

        Returns
        -------
        collections : `dict` of `str` : `int`
            The sizes of the collections for the given dataset type, sorted by size descending.
        """
        butler = dafButler.Butler(self.repo)

        # Determine the collections to query
        if collection_names is None and collection_regex is None:
            raise ValueError("Must specify either collection_names or collection_regex")
        if collection_names is None:
            all_collections = butler.registry.queryCollections()
            collection_names = [
                col for col in all_collections if re.match(collection_regex, col)
            ]

        # Get the dataset sizes for each collection.
        collections = {}
        for collection in collection_names:
            collections[collection] = butler.registry.queryDatasets(
                datasetType, collections=[collection]
            ).count()

        return {
            k: v
            for k, v in sorted(collections.items(), key=lambda x: x[1], reverse=True)
        }

    def ingest_collections(
        self,
        collections,
        datasetType,
        is_parallel=True,
        n_workers=8,
        output_dir=None,
        naming_scheme=None,
        overwrite=False,
    ):
        """
        Ingest the data from the Butler into an ImageCollection.

        Parameters
        ----------
        collections : `list[str]`
            The collections to ingest.
        datasetType : `str`
            The dataset type to ingest.
        is_parallel : `bool`, optional
            If True, use parallel processing. Default is True.
        n_workers : `int`, optional
            The number of workers to use in parallel processing. If None, attempts to choose the number of workers by OS provided CPU count. Default is 8.
        output_dir : `str`, optional
            The directory to write the ImageCollections to. If None, images are not written to disk.
        naming_scheme: `callable`, optional
            A function to use to name the output ImageCollection. Default is None.
        overwrite : `bool`, optional
            If True, overwrite existing ImageCollections. Default is False.

        Returns
        -------
        results : `list[ImageCollection]`
            The list of ImageCollections created from each Butler collection.
        """
        repo = self.repo
        results = []
        if not is_parallel:
            for collection_name in collections:
                logger.debug(f"Ingesting {collection_name}")
                results.append(
                    _ingest_collection(
                        repo,
                        collection_name,
                        datasetType,
                        output_dir,
                        naming_scheme,
                        overwrite,
                    )
                )
        else:
            if n_workers is None:
                # Note: that this will use all available CPUs, which may be an unreasonable number on some systems.
                n_workers = os.cpu_count()
            # Process each Butler collection in parallel.
            with ProcessPoolExecutor(n_workers) as executor:
                futures = []
                for collection_name in collections:
                    futures.append(
                        executor.submit(
                            _fake_ingest_collection,  # TODO fix me.
                            repo,
                            collection_name,
                            datasetType,
                            output_dir=output_dir,
                            naming_scheme=naming_scheme,
                            overwrite=overwrite,
                        )
                    )

                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        # Note that since waiting for the future's result is blocking, the time of
                        # the exception's log here be later than when it actually occurred.
                        logger.error(f"Error ingesting collection: {e}")
        return results
