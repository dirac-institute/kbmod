import lsst
import lsst.daf.butler as dafButler
import lsst.sphgeom as sphgeom

import os
import glob

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import time
from dateutil import parser

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from astropy.time import Time  # for converting Butler visitInfo.date (TAI) to UTC strings
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits

import pickle

def _chunked_dataIds(dataIds, chunk_size=200):
    """Helper function to yield successive chunk_size chunks from dataIds."""
    for i in range(0, len(dataIds), chunk_size):
        yield dataIds[i : i + chunk_size]

# A class for RegionSearch that takes an LSST butler repo path or an LSST butler object
class RegionSearch:
    def __init__(self, repo_path, collections, dataset_types, butler=None, parallel=True):
        """
        Parameters
        ----------
        repo_path : str
            The path to the LSST butler repository.    
        butler : `lsst.daf.butler.Butler`
            The butler object to use for data access.
        """
        self.repo_path = repo_path
        if butler is not None:
            self.butler = butler
        else:
            self.butler = dafButler.Butler(self.repo_path)
        
        self.desired_collections = collections
        self.desired_dataset_types = dataset_types

        self.parallel = parallel

    def new_butler(self):
        return dafButler.Butler(self.repo_path)

    # TODO make static method
    def get_collection_names(self):
        """
        Returns
        -------
        list of str
            The list of available collections in the butler repository.
        """
        # TODO consider adding caching for the queried collections
        return self.butler.registry.queryCollections()
    

    def set_desired_collections(self, desired_collections):
        """
        Parameters
        ----------
        desired_collections : list of str
            The list of desired collections to use for the region search.
        """
        self.desired_collections = desired_collections

    def get_dataset_type_freq(self, collections=None):
        """
        Parameters
        ----------
        collections : list(str)
            The names of the collection to get the dataset type stats for. If None, use self.desired_collections if availabel.
        
        Returns
        -------
        dict
            A dictionary of freq for the given collection.
        """
        if collections is None and self.desired_collections is not None:
            collections = self.desired_collections
        freq = {}
        
        # Iterate over all dataset types and count the number of freq
        for dt in self.butler.queryDatasetTypes():
            refs = []
            if collections:
                refs = self.butler.registry.queryDatasets(dt, collections=self.desired_collections)
            else:
                refs = self.butler.registry.queryDatasets(dt)
            if dt.name not in freq:
                freq[dt.name] = 0
            freq[dt.name] += len(refs)

        return freq
    
    def set_desired_dataset_types(self, desired_dataset_types):
        """
        Set the desired dataset types to use for the region search.
        """
        self.set_desired_dataset_types = desired_dataset_types

    def get_vdr_data(self, collections=None, dataset_types=None):
        """
        Constructs the VDR (Visit Detector Region) data for the given collections and dataset types.

        VDRs are the regions of the detector that are covered by a visit. They contain what we need in terms of
        regions hashes and unique dataIds.
        
        Parameters
        ----------
        collections : list(str)
            The names of the collection to get the dataset type stats for. If None, use self.desired_collections.
        dataset_types : list(str)
            The names of the dataset types to get the dataset type stats for. If None, use self.desired_dataset_types.
        """
        if not collections:
            if not self.desired_collections:
                raise ValueError("No collections specified")
            collections = self.desired_collections
        
        if not dataset_types:
            if not self.desired_dataset_types:
                raise ValueError("No dataset types specified")
            
        # TODO should we use an Object Instead? What's the best way to serialize
        vdr_dict = {"data_id": [], "region": [], "detector": []}

        for dt in dataset_types:
            refs = self.butler.registry.queryDimensionRecords(
                "visit_detector_region", datasets=dt, collections=collections
            )
            for ref in refs:
                detector = ref.dataId["detector"]
                vdr_dict["data_id"].append(ref.dataId)
                vdr_dict["region"].append(ref.region)
                vdr_dict["detector"].append(ref.detector)
            
        return vdr_dict

    def get_instruments(self, vdr_ids, butler=None, collections=None, first_instrument_only=True):
        """
        Get the instruments for the given VDR dataIds.
        
        Parameters
        ----------
        vdr_ids : list(dict)
            The list of VDR dataIds to get the instruments for.
        first_instrument_only : bool
            If True, return only the first instrument we find.
        """
        instruments = []
        if collections is None:
            if self.desired_collections is None:
                raise ValueError("No collections specified")
            collections = self.desired_collections
        
        for vdr_id in vdr_ids:
            instrument = self.butler.get("calexp.visitInfo", dataId=vdr_id, collections=collections)
            if first_instrument_only:
                return [instrument]
            instruments.append(instrument)
        return instruments

    def set_desired_instruments(self, instruments):
        return self.desired_instruments(instruments)
    

def get_uris_serial(self, dataIds, dataset_types=None, collections=None, butler=None):
    """Fetch URIs for a list of dataIds."""
    if butler is None:
        butler = self.butler
    results = []
    if dataset_types is None:
        if self.desired_dataset_types is None:
            raise ValueError("No dataset types specified")
        dataset_types = self.desired_dataset_types
    if collections is None:
        if self.desired_collections is None:
            raise ValueError("No collections specified")
        collections = self.desired_collections

    for dataId in dataIds:
        try:
            uri = self.butler.getURI(dataset_types[0], dataId=dataId, collections=collections)
            uri = uri.geturl()  # Convert to URL string
            results.append(uri)
        except Exception as e:
            print(f"Failed to retrieve path for dataId {dataId}: {e}")
    return results

def get_uris(self, dataIds, dataset_types=None, collections=None):
    if dataset_types is None:
        if self.desired_dataset_types is None:
            raise ValueError("No dataset types specified")
        dataset_types = self.desired_dataset_types
    if collections is None:
        if self.desired_collections is None:
            raise ValueError("No collections specified")
        collection = self.desired_collections

    if not self.parallel:
        return self.get_uris_serial(dataIds, dataset_types, collections)

    # Divide the dataIds into chunks to be processed in parallel
    dataIds_chunks = list(_chunked_dataIds(dataIds))

    # Use a ProcessPoolExecutor to fetch URIs in parallel
    result_uris = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(self.get_uris_from_data_ids, chunk, self.new_butler(), dataset_types, collections) for chunk in dataIds_chunks]
        for future in as_completed(futures):
            result_uris.extend(future.result())

    return result_uris


def get_center_ra_dec(self, region):
    """
    We pull the 2D boundingBox (not the boundingBox3d) from a region.
    Then we extract the center's (RA, Dec) coordinates.
    """
    bbox_center = region.getBoundingBox().getCenter()
    ra = bbox_center.getLon().asDegrees()
    dec = bbox_center.getLat().asDegrees()
    return ra, dec


def get_timestamps_serial(self, dataIds, butler=None):
    """
    Get the timestamps for the given dataIds.
    """
    if butler is None:
        butler = self.butler

    timestamps = []
    for dataId in dataIds:
        visit_info = butler.get("calexp.visitInfo", dataId=dataId, collections=self.desired_collections)
        # TODO how do we want to represent the timestamps
        t = Time(str(visit_info.date).split('"')[1], format="isot", scale="tai")
        timestamps.append(str(t.utc))
    return timestamps

def get_timestamps(self, dataIds):
    """
    Get the timestamps for the given dataIds in parallel.
    """
    if not self.parallel:
        return self.get_timestamps_serial(dataIds)

    # Divide the dataIds into chunks to be processed in parallel
    dataIds_chunks = list(_chunked_dataIds(dataIds))

    # Use a ProcessPoolExecutor to fetch timestamps in parallel
    timestamps = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(self.get_timestamps, chunk, self.new_butler()) for chunk in dataIds_chunks]
        for future in as_completed(futures):
            timestamps.extend(future.result())

    return timestamps

def find_overlapping_coords(self, data, uncertainty_radius=30):
    """
    Find the overlapping coordinates in the given data.
    """
    # TODO consider adding caching for the overlapping sets
    overlapping_sets = {}

    # TODO should data be a dataframe?

    # Assuming uncertainty_radius is provided as a float in arcseconds
    uncertainty_radius_as = uncertainty_radius * u.arcsec

    all_coords = SkyCoord(
        ra=[x[0] for x in data["center_coord"]] * u.degree,
        dec=[x[1] for x in data["center_coord"]] * u.degree,
    )

    overlapping_sets = {}
    set_counter = 1
    processed_data_ids = [] # TODO do we need this?

    for index, coord in enumerate(all_coords):
        data_id = data["data_id"][index]
        if data_id not in processed_data_ids:
            distances = coord.separation(all_coords).to(u.arcsec).value
            
            # Perform comparison as numeric values, bypassing direct unit comparison
            within_radius = (distances <= uncertainty_radius_as.value) & (distances > 0)

            if any(within_radius):
                overlapping_indices = [
                    i
                    for i, distance in enumerate(distances)
                    if (distance <= uncertainty_radius_as.value) and i != index
                ]
                overlapping_data_ids = data["data_id"][overlapping_indices].tolist()
                overlapping_data_ids.append(data_id)

                processed_data_ids.extend(overlapping_data_ids)

                overlapping_sets[f"set_{set_counter}"] = overlapping_data_ids
                set_counter += 1

    return overlapping_sets

def retrieve_image_sets(self, overlap_uncertainty_radius_arcsec=30):
    """
    """
    data = self.get_vdr_data(collections=self.desired_collections, dataset_types=self.desired_dataset_types)
    data["center_coord"] = [self.get_center_ra_dec(region) for region in data["region"]]
    data["uri"] = self.get_uris(data["data_id"])

    data["ut"] = self.get_timestamps(dataIds=data["data_id"])
    data["ut_datetime"] = pd.to_datetime(data["ut"])
    overlapping_sets = find_overlapping_coords(df=data)
    return data, overlapping_sets