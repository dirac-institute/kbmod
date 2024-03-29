try:
    import lsst.daf.butler as dafButler
except ImportError:
    raise ImportError("LSST stack not found. Please install the LSST stack to use this module.")

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from astropy.table import Table


def _chunked_data_ids(dataIds, chunk_size=200):
    """Helper function to yield successive chunk_size chunks from dataIds."""
    for i in range(0, len(dataIds), chunk_size):
        yield dataIds[i : i + chunk_size]


"""
A class for searching through a dataset for data suitable for KBMOD processing,

With a path to a butler repository, it provides helper methods for basic exploration of the data,
methods for retrieving data from the butler for search, and transformation of the data
into a KBMOD ImageCollection for further processing.

Note that currently we store results from the butler in an Astropy Table. In the future,
we will likely want to use a database for faster performance and to handle processing of
datasets that are too large to fit in memory.
"""


class RegionSearch:
    def __init__(
        self,
        repo_path,
        collections,
        dataset_types,
        butler=None,
        visit_info_str="Exposure.visitInfo",
        parallel=False,
        fetch_data=False,
    ):
        """
        Parameters
        ----------
        repo_path : str
            The path to the LSST butler repository.
        collections : list of str
            The list of desired collection names within the Butler repository`
        dataset_types : list of str
            The list of desired dataset types within the Butler repository.
        butler : `lsst.daf.butler.Butler`, optional
            The Butler object to use for data access. If None, a new Butler object will be created from `repo_path`.
        visit_info_str : str
            The name used when querying the butler for VisitInfo for exposures. Default is "Exposure.visitInfo".
        parallel : bool
            If True, use parallel processing where possible. Note that each parallel worker
            will instantiate its own Butler objects, Default is False.
        fetch_data: bool, optional
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
        self.parallel = parallel

        # Create an empty table to store the VDR (Visit, Detector, Region) data from the butler.
        self.vdr_data = Table()
        if fetch_data:
            # Fetch the VDR data from the butler
            self.vdr_data = self.fetch_vdr_data()

    @staticmethod
    def get_collection_names(butler=None, repo_path=None):
        """
        Get the list of the names of available collections in a butler repository.
        Parameters
        ----------
        butler | repo_path : `lsst.daf.butler.Butler` | str
            The Butler object or a path to the LSST butler repository from which to create a butler.
        Returns
        -------
        list of str
            The list of available collections in the butler repository.
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
        collections : list of str, optional
            The names of collections from which we can querry the dataset type frequencies. If None, use all collections.
        Returns
        -------
        dict
            A dictionary of frequency of refs per dataset type in the given collections.
        """
        if butler is None:
            if repo_path is None:
                raise ValueError("Must specify one of repo_path or butler")
            butler = dafButler.Butler(repo_path)

        # Iterate over all dataset types and count the frequency of refs associated with each
        freq = {}
        for dt in butler.registry.queryDatasetTypes():
            refs = None
            if collections:
                refs = butler.registry.queryDatasets(dt, collections=collections)
            else:
                refs = butler.registry.queryDatasets(dt)
            if refs is not None:
                if dt.name not in freq:
                    freq[dt.name] = 0
                freq[dt.name] += refs.count(exact=True, discard=True)

        return freq

    def new_butler(self):
        """Instantiates a new Butler object from the repo_path."""
        return dafButler.Butler(self.repo_path)

    def set_collections(self, collections):
        """
        Set which collections to use when querying data from the butler.

        Parameters
        ----------
        collections : list of str
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
        collections : list(str)
            The names of the collection to get the dataset type stats for. If None, use self.collections.
        dataset_types : list(str)
            The names of the dataset types to get the dataset type stats for. If None, use self.dataset_types.
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
        return Table(vdr_dict)

    def get_instruments(self, data_ids=None, first_instrument_only=False):
        """
        Get the instruments for the given VDR data ids.

        Parameters
        ----------
        data_ids : list(dict), optional
            The list of VDR data IDs to get the instruments for. By default uses previously fetched data_ids
        first_instrument_only : bool
            If True, return only the first instrument we find.
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

    def get_uris_serial(self, data_ids, dataset_types=None, collections=None, butler=None):
        """Fetch URIs for a list of dataIds in serial fashion.

        Parameters
        ----------
        data_ids : list(dict)
            The list of data Ids to fetch URIs for.
        dataset_types : list(str)
            The dataset types to use when fetching URIs. If None, use self.dataset_types.
        collections : list(str)
            The collections to use when fetching URIs. If None, use self.collections.
        butler : `lsst.daf.butler.Butler`, optional
            The Butler object to use for data access. If None, use self.butler.

        Returns
        -------
        list(str)
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

        results = []
        for data_id in data_ids:
            try:
                uri = self.butler.getURI(dataset_types[0], dataId=data_id, collections=collections)
                uri = uri.geturl()  # Convert to URL string
                results.append(uri)
            except Exception as e:
                print(f"Failed to retrieve path for dataId {data_id}: {e}")
        return results

    def get_uris(self, data_ids, dataset_types=None, collections=None):
        """
        Get the URIs for the given dataIds.

        Parameters
        ----------
        data_ids : list(dict)
            The list of data Ids to fetch URIs for.
        dataset_types : list(str)
            The dataset types to use when fetching URIs. If None, use self.dataset_types.
        collections : list(str)
            The collections to use when fetching URIs. If None, use self.collections.

        Returns
        -------
        list(str)
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

        if not self.parallel:
            return self.get_uris_serial(data_ids, dataset_types, collections)

        # Divide the data_ids into chunks to be processed in parallel
        data_id_chunks = list(_chunked_data_ids(data_ids))

        # Use a ProcessPoolExecutor to fetch URIs in parallel
        result_uris = []
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [
                executor.submit(
                    self.get_uris_serial,
                    chunk,
                    dataset_types=dataset_types,
                    collections=collections,
                    butler=self.new_butler(),
                )
                for chunk in data_id_chunks
            ]
            for future in as_completed(futures):
                result_uris.extend(future.result())

        return result_uris

    def get_center_ra_dec(self, region):
        """
        Get the center RA and Dec for the given region.
        """
        # Note we get the 2D boundingBox (not the boundingBox3d) from a region.
        # We then extract the RA and Dec from the center of the bounding box.
        bbox_center = region.getBoundingBox().getCenter()
        ra = bbox_center.getLon().asDegrees()
        dec = bbox_center.getLat().asDegrees()
        return ra, dec
