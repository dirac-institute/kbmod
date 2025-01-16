try:
    import lsst.daf.butler as dafButler
except ImportError:
    raise ImportError("LSST stack not found. Please install the LSST stack to use this module.")

from concurrent.futures import ProcessPoolExecutor, as_completed

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

def _ingest_collection(repo, collection_name, datasetType, output_dir, overwrite=False):
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
        The directory to write the ImageCollections to.
    overwrite : `bool`, optional
        If True, overwrite existing ImageCollections. Default is False.

    Returns
    -------
    None
    """

    output_path = os.path.join(output_dir, f"{collection_name}.collection")
    if not overwrite and os.path.exists(output_path):
        #logger.debug(f"Skipping {collection_name} as it already exists.")
        return

    butler = dafButler.Butler(repo)
    refs = butler.registry.queryDatasets(datasetType, collections=[collection_name])
    if not refs:
        #logger.debug(f"No datasets found for {datasetType} in {collection_name}.")
        return

    ic = kbmod.ImageCollection.fromTargets(
        refs,
        butler=butler,
        force="ButlerStandardizer"
    )

    ic["collection"] = collection_name

    if not overwrite and os.path.exists(output_path):
        #logger.debug(f"Output path {output_path} was created while processing {collection_name}. Aborting.")
        return
    ic.write(output_path, overwrite=overwrite)

    bbox_cols = [
        "ra", "dec", "ra_tl", "dec_tl", "ra_tr", "dec_tr",
        "ra_bl", "dec_bl", "ra_br", "dec_br", "mjd_mid", "dataId"
    ]
    bbox_path = os.path.join(output_dir, f"{collection_name}.bbox")
    ic[bbox_cols].write(bbox_path, format="ascii.ecsv",
                    overwrite=True, # Always overwrite if we wrote new ImageCollection
    )

    #logger.info(f"Finished ingesting {collection_name}.")

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

    def get_butler_collections(self, datasetType, collectionNamePattern):
        """
        Get the collections that match the given dataset type and collection name pattern.

        Note collections are an ordered dictionary of how many of the dataset size for
        each butler collection.

        Parameters
        ----------
        datasetType : `str`
            The dataset type to search for.
        collectionNamePattern : `str`
            The pattern to match for collection names.

        Returns
        -------
        collections : `OrderedDict` of `str` : `int`
            The collections that match the given dataset type and collection name pattern.
        """
        butler = dafButler.Butler(self.repo)
        all_collections = butler.registry.queryCollections()
        logger.debug(f"Found {len(all_collections)} collections in repository {self.repo}.")

        # Match for the subset of collections that match our regex.
        pattern = re.compile(collectionNamePattern)
        matches = (re.match(pattern, c) for c in all_collections)

        # Filter out the None matches and get the collection names.
        collection_names = [m.group() for m in matches if m is not None]

        # Get the dataset sizes for each collection.
        collections = {}
        for collection in collection_names:
            collections[collection] = butler.registry.queryDatasets(datasetType, collections=[collection]).count()

        return {k: v for k, v in sorted(collections.items(), key=lambda x: x[1], reverse=True)}



    def ingest_collections(self, collections, datasetType, is_parallel=True, n_workers=None, output_dir=None, overwrite=False):
        """
        Ingest the data from the Butler into an ImageCollection.

        Parameters
        ----------
        collections : `list[str]`
            The collections to ingest.
        is_parallel : `bool`, optional
            If True, use parallel processing. Default is True.
        output_dir : `str`, optional
            The dataset type to ingest.
        output_dir : `str`, optional
            The directory to write the ImageCollections to. If None, images are not written to disk.
        overwrite : `bool`, optional
            If True, overwrite existing ImageCollections. Default is False.

        Returns
        -------
        None
        """
        repo = self.repo
        if not is_parallel:
            for collection_name in collections:
                _ingest_collection(repo, collection_name, datasetType, output_dir, overwrite)
        else:
            if n_workers is None:
                n_workers = os.cpu_count()
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for collection_name in collections:
                    futures.append(executor.submit(
                        repo,
                        _ingest_collection,
                        collection_name,
                        datasetType,
                        output_dir,
                        overwrite))

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error ingesting collection: {e}")


class RegionSearch:
    """
    A class for searching through a dataset for data suitable for KBMOD processing,

    With a path to a butler repository, it provides helper methods for basic exploration of the data,
    methods for retrieving data from the butler for search, and transformation of the data
    into a KBMOD ImageCollection for further processing.

    Note that currently we store results from the butler in an Astropy Table. In the future,
    we will likely want to use a database for faster performance and to handle processing of
    datasets that are too large to fit in memory.
    """

    def __init__(
        self,
        repo_path,
        collections,
        dataset_types,
        butler=None,
        visit_info_str="Exposure.visitInfo",
        max_workers=None,
        fetch_data_on_start=False,
    ):
        """
        Parameters
        ----------
        repo_path : `str`
            The path to the LSST butler repository.
        collections : `list[str]`
            The list of desired collection names within the Butler repository`
        dataset_types : `list[str]`
            The list of desired dataset types within the Butler repository.
        butler : `lsst.daf.butler.Butler`, optional
            The Butler object to use for data access. If None, a new Butler object will be created from `repo_path`.
        visit_info_str : `str`
            The name used when querying the butler for VisitInfo for exposures. Default is "Exposure.visitInfo".
        max_workers : `int`, optional
            The maximum number of workers to use in parallel processing. Note that each parallel worker will instantiate its own Butler
            objects. If not provided, parallel processing is disabled.
        fetch_data_on_start: `bool`, optional
            If True, fetch the VDR data when the object is created. Default is True.
        """
        self.repo_path = repo_path
        if butler is not None:
            self.butler = butler
        else:
            self.butler = dafButler.Butler(self.repo_path)

        self.collections = collections
        self.dataset_types = dataset_types
        self.visit_info_str = visit_info_str
        self.max_workers = max_workers

        # Create an empty table to store the VDR (Visit, Detector, Region) data from the butler.
        self.vdr_data = Table()
        if fetch_data_on_start:
            # Fetch the VDR data from the butler
            self.vdr_data = self.fetch_vdr_data()

    @staticmethod
    def get_collection_names(butler=None, repo_path=None):
        """
        Get the list of the names of available collections in a butler repository.
        Parameters
        ----------
        butler | repo_path : `lsst.daf.butler.Butler` | `str`
            The Butler object or a path to the LSST butler repository from which to create a butler.
        Returns
        -------
        collections : `list[str]`
            The list of the names of available collections in the butler repository.
        """
        if butler is None:
            if repo_path is None:
                raise ValueError("Must specify one of repo_path or butler")
            butler = dafButler.Butler(repo_path)
        return butler.registry.queryCollections()

    @staticmethod
    def get_dataset_type_freq(butler=None, repo_path=None, collections=None):
        """
        Get the frequency of refs per dataset types across the given collections.

        Parameters
        ----------
        butler | repo_path : `lsst.daf.butler.Butler` | str
            The Butler object or a path to the LSST butler repository from which to create a butler.
        collections : `list[str]`, optional
            The names of collections from which we can querry the dataset type frequencies. If None, use all collections.
        Returns
        -------
        ref_freq : `dict`
            A dictionary of frequency of refs per dataset type in the given collections.
        """
        if butler is None:
            if repo_path is None:
                raise ValueError("Must specify one of repo_path or butler")
            butler = dafButler.Butler(repo_path)

        # Iterate over all dataset types and count the frequency of refs associated with each
        ref_freq = {}
        for dt in butler.registry.queryDatasetTypes():
            refs = None
            if collections:
                refs = butler.registry.queryDatasets(dt, collections=collections)
            else:
                refs = butler.registry.queryDatasets(dt)
            if refs is not None:
                if dt.name not in ref_freq:
                    ref_freq[dt.name] = 0
                ref_freq[dt.name] += refs.count(exact=True, discard=True)

        return ref_freq

    def is_parallel(self):
        """Returns True if parallel processing was requested."""
        return self.max_workers is not None

    def new_butler(self):
        """Instantiates a new Butler object from the repo_path."""
        if self.butler is not None:
            return dafButler.Butler(self.repo_path, registry=self.butler.registry)
        return dafButler.Butler(self.repo_path)

    def set_collections(self, collections):
        """
        Set which collections to use when querying data from the butler.

        Parameters
        ----------
        collections : `list[str]`
            The list of desired collections to use for the region search.
        """
        self.collections = collections

    def set_dataset_types(self, dataset_types):
        """
        Set the desired dataset types to use when querying the butler.
        """
        self.dataset_types = dataset_types

    def get_vdr_data(self):
        """Returns the VDR data"""
        return self.vdr_data

    def fetch_vdr_data(self, collections=None, dataset_types=None):
        """
        Fetches the VDR (Visit, Detector, Region) data for the given collections and dataset types.

        VDRs are the regions of the detector that are covered by a visit. They contain what we need in terms of
        regions hashes and unique dataIds.

        Parameters
        ----------
        collections : `list[str]`
            The names of the collection to get the dataset type stats for. If None, use self.collections.
        dataset_types : `list[str]`
            The names of the dataset types to get the dataset type stats for. If None, use self.dataset_types.

        Returns
        -------
        vdr_data : `astropy.table.Table`
            An Astropy Table containing the VDR data and associated URIs and RA/Dec center coordinates.
        """
        if not collections:
            if not self.collections:
                raise ValueError("No collections specified")
            collections = self.collections

        if not dataset_types:
            if not self.dataset_types:
                raise ValueError("No dataset types specified")
            dataset_types = self.dataset_types

        vdr_dict = {"data_id": [], "region": [], "detector": [], "uri": [], "center_coord": []}

        for dt in dataset_types:
            refs = self.butler.registry.queryDimensionRecords(
                "visit_detector_region", datasets=dt, collections=collections
            )
            for ref in refs:
                vdr_dict["data_id"].append(ref.dataId)
                vdr_dict["region"].append(ref.region)
                vdr_dict["detector"].append(ref.detector)
                vdr_dict["center_coord"].append(self.get_center_ra_dec(ref.region))

        # Now that we have the initial VDR data ids, we can also fetch the associated URIs
        vdr_dict["uri"] = self.get_uris(vdr_dict["data_id"])

        # return as an Astropy Table
        self.vdr_data = Table(vdr_dict)

        return self.vdr_data

    def get_instruments(self, data_ids=None, first_instrument_only=False):
        """
        Get the instruments for the given VDR data ids.

        Parameters
        ----------
        data_ids : `iterable(dict)`, optional
            A collection of VDR data IDs to get the instruments for. By default uses previously fetched data_ids
        first_instrument_only : `bool`, optional
            If True, return only the first instrument we find. Default is False.

        Returns
        -------
        instruments : `list`
            A list of instrument objects for the given data IDs.
        """
        if data_ids is None:
            data_ids = self.vdr_data["data_id"]

        instruments = []
        for data_id in data_ids:
            instrument = self.butler.get(self.visit_info_str, dataId=data_id, collections=self.collections)
            if first_instrument_only:
                return [instrument]
            instruments.append(instrument)
        return instruments

    def _get_uris_serial(
        self, data_ids, dataset_types=None, collections=None, butler=None, trim_uri_func=_trim_uri
    ):
        """Fetch URIs for a list of dataIds in serial fashion.

        Parameters
        ----------
        data_ids : `iterable(dict)`
            A collection of data Ids to fetch URIs for.
        dataset_types : `list[str]`
            The dataset types to use when fetching URIs. If None, use self.dataset_types.
        collections : `list[str]`
            The collections to use when fetching URIs. If None, use self.collections.
        butler : `lsst.daf.butler.Butler`, optional
            The Butler object to use for data access. If None, use self.butler.
        trim_uri_func: `function`, optional
            A function to trim the URIs. Default is _trim_uri.

        Returns
        -------
        uris : `list[str]`
            The list of URIs for the given data Ids.
        """
        if butler is None:
            butler = self.butler
        if dataset_types is None:
            if self.dataset_types is None:
                raise ValueError("No dataset types specified")
            dataset_types = self.dataset_types
        if collections is None:
            if self.collections is None:
                raise ValueError("No collections specified")
            collections = self.collections

        uris = []
        for data_id in data_ids:
            try:
                uri = self.butler.getURI(dataset_types[0], dataId=data_id, collections=collections)
                uri = uri.geturl()  # Convert to URL string
                uris.append(trim_uri_func(uri))
            except Exception as e:
                print(f"Failed to retrieve path for dataId {data_id}: {e}")
        return uris

    def get_uris(self, data_ids, dataset_types=None, collections=None, trim_uri_func=_trim_uri):
        """
        Get the URIs for the given dataIds.

        Parameters
        ----------
        data_ids : `iterable(dict)`
            A collection of data Ids to fetch URIs for.
        dataset_types : `list[str]`
            The dataset types to use when fetching URIs. If None, use self.dataset_types.
        collections : `list[str]`
            The collections to use when fetching URIs. If None, use self.collections.
        trim_uri_func: `function`, optional
            A function to trim the URIs. Default is _trim_uri.

        Returns
        -------
        uris : `list[str]`
            The list of URIs for the given data Ids.
        """
        if dataset_types is None:
            if self.dataset_types is None:
                raise ValueError("No dataset types specified")
            dataset_types = self.dataset_types
        if collections is None:
            if self.collections is None:
                raise ValueError("No collections specified")
            collections = self.collections

        if not self.is_parallel():
            return self._get_uris_serial(data_ids, dataset_types, collections)

        # Divide the data_ids into chunks to be processed in parallel
        data_id_chunks = list(_chunked_data_ids(data_ids))

        # Use a ProcessPoolExecutor to fetch URIs in parallel
        uris = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._get_uris_serial,
                    chunk,
                    dataset_types=dataset_types,
                    collections=collections,
                    butler=self.new_butler(),
                    trim_uri_func=trim_uri_func,
                )
                for chunk in data_id_chunks
            ]
            for future in as_completed(futures):
                uris.extend(future.result())

        return uris

    def get_center_ra_dec(self, region):
        """
        Get the center RA and Dec for the given region.

        Parameters
        ----------
        region : `lsst::sphgeom::Region Class Reference`
            The region for which to get the center RA and Dec.

        Returns
        -------
        ra, dec : `float`, `float`
            The center RA and Dec in degrees.
        """
        # Note we get the 2D boundingBox (not the boundingBox3d) from a region.
        # We then extract the RA and Dec from the center of the bounding box.
        bbox_center = region.getBoundingBox().getCenter()
        ra = bbox_center.getLon().asDegrees()
        dec = bbox_center.getLat().asDegrees()
        return ra, dec

    def find_overlapping_coords(self, data=None, uncertainty_radius=30):
        """
        Find the overlapping sets of data based on the center coordinates of the data.

        Parameters
        ----------
        data : `astropy.table.Table`, optional
            The data to search for overlapping sets. If not provided, use the VDR data.

        uncertainty_radius : `float`
            The radius in arcseconds to use when determining if two data points overlap.

        Returns
        -------
        overlapping_sets : list[list[int]]
            A list of overlapping sets of data. Each set is a list of the indices within
            the VDR (Visit, Detector, Region) table.
        """
        if not data:
            if len(self.vdr_data) == 0:
                self.vdr_data = self.fetch_vdr_data()
            data = self.vdr_data

        # Assuming uncertainty_radius is provided as a float in arcseconds
        uncertainty_radius_as = uncertainty_radius * u.arcsec

        # Convert the center coordinates to SkyCoord objects
        all_ra_dec = SkyCoord(
            ra=[x[0] for x in data["center_coord"]] * u.degree,
            dec=[x[1] for x in data["center_coord"]] * u.degree,
        )

        # Indices of the data ids that we have already processed
        processed_data_ids = set([])
        overlapping_sets = []
        for i in range(len(all_ra_dec) - 1):
            coord = all_ra_dec[i]
            if i not in processed_data_ids:
                # We haven't chosen the current index for a previous pile, which means
                # that it was not within the separation distance of any earlier coordinate
                # with an index less than 'i'. So we only have to compute the separation
                # distances for the coordinates that come after 'i'.
                distances = coord.separation(all_ra_dec[i + 1 :]).to(u.arcsec).value

                # Consider choosing the the current index.
                overlapping_data_ids = [i]

                for j in range(len(distances)):
                    if distances[j] <= uncertainty_radius_as.value:
                        # We add the indices of other coordinates within the radius,
                        # offset by our starting 'all_ra_dec' index of i + 1
                        overlapping_data_ids.append(i + 1 + j)
                if len(overlapping_data_ids) > 1:
                    # Add our choice of overlapping set to our results.
                    processed_data_ids.update(overlapping_data_ids)
                    overlapping_sets.append(overlapping_data_ids)

        return overlapping_sets
