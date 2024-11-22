import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky

import kbmod.search as kb
from kbmod.trajectory_utils import trajectory_predict_skypos
from collections import Counter

logger = kb.Logging.getLogger(__name__)


class KnownObjsMatcher:
    """
    A class which ingests an astopy table of object data expected to be found in the dataset
    searched by KBMOD (either real objects or inserted synthetic fakes) and provides methods for
    matching to the observations in a given set of KBMOD Results.

    It allows for configuration of how the matching is done, including the maximum
    separation in arcseconds between a known object and a result to be considered a match,
    the maximum time separation in seconds between a known object and the observation
    used in a KBMOD result.

    In addition to modifying a KBMOD `Results` table to include columns for matched known objects,
    it also provides methods for filtering the results based on the matches. This includes
    marking observations that matched to known objects as invalid, and filtering out results that matched to
    known objects by either the minimum number of observations that matched to that known object or the
    proportion of observations from the catalog for that known object that were matched to a given result.
    """

    def __init__(
        self,
        table,
        obstimes,
        matcher_name,
        sep_thresh=1.0,
        time_thresh_s=600.0,
        mjd_col="mjd_mid",
        ra_col="RA",
        dec_col="DEC",
        name_col="Name",
    ):
        """
        Parameters
        ----------
        table : astropy.table.Table
            A table containing our catalog of observations of known objects.
        obstimes : list(float)
            The MJD times of each observation within KBMOD results we want to match to
            the known objects.
        matcher_name : str
            The name of the filter to apply to the results. This both determines
            the name of the column of matched observations which may be added to
            the `Results` table and how the filtering and matching phases are identified within KBMOD logs.
        sep_thresh : float, optional
            The maximum separation in arcseconds between a known object and a result
            to be considered a match. Default is 1.0.
        time_thresh_s : float, optional
            The maximum time separation in seconds between a known object and the observation
            used in a KBMOD result. Default is 600.0.
        mjd_col : str, optional
            The name of the catalog column containing the MJD of the known objects. Default is "mjd_mid".
        ra_col : str, optional
            The name of the catalog column containing the RA of the known objects. Default is "RA".
        dec_col : str, optional
            The name of the catalog column containing the DEC of the known objects. Default is "DEC".
        name_col : str, optional
            The name of the catalog column containing the name of the known objects. Default is "Name".

        Raises
        ------
        ValueError
            If the required columns are not present in the table.

        Returns
        -------
        KnownObjsMatcher
            A KnownObjsMatcher object.
        """
        self.data = table

        # Map our required columns to any specified column names.
        self.mjd_col = mjd_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.name_col = name_col

        # Check that the required columns are present
        user_cols = set([self.mjd_col, self.ra_col, self.dec_col, self.name_col])
        invalid_cols = user_cols - set(self.data.colnames)
        if invalid_cols:
            raise ValueError(f"{invalid_cols} not found in KnownObjs data.")

        self.obstimes = obstimes
        if len(self.obstimes) == 0:
            raise ValueError("No obstimes provided")

        self.matcher_name = matcher_name
        self.sep_thresh = sep_thresh * u.arcsec
        self.time_thresh_s = time_thresh_s

        # Pre-filter down our data to window of temporally relevant observations to speed up matching.
        time_thresh_days = self.time_thresh_s / (24 * 3600)  # Convert seconds to days
        start_mjd = max(0, min(self.obstimes) - time_thresh_days - 1e-6)
        end_mjd = max(self.obstimes) + time_thresh_days + 1e-6

        # Filter out known object observations outside of our time thresholds
        self.data = self.data[(self.data[self.mjd_col] >= start_mjd) & (self.data[self.mjd_col] <= end_mjd)]

    def match_min_obs_col(self, min_obs):
        """A colummn name for objects that matched results based on the minimum number of observations."""
        return f"recovered_{self.matcher_name}_min_obs_{min_obs}"

    def match_obs_ratio_col(self, obs_ratio):
        # A column name for objects that matched results based on the proportion of observations that
        # matched to the known observations for that object within the catalog.
        return f"recovered_{self.matcher_name}_obs_ratio_{obs_ratio}"

    def __len__(self):
        """Returns the number of observations known objects of interest in this matcher's catalog."""
        return len(self.data)

    def get_mjd(self, ko_idx):
        """
        Returns the MJD of the known object at a given index.
        """
        return self.data[ko_idx][self.mjd_col]

    def get_ra(self, ko_idx):
        """
        Returns the RA of the known object at a given index.
        """
        return self.data[ko_idx][self.ra_col]

    def get_dec(self, ko_idx):
        """
        Returns the DEC of the known object at a given index.
        """
        return self.data[ko_idx][self.dec_col]

    def get_name(self, ko_idx):
        """
        Returns the name of the known object at a given index.
        """
        return self.data[ko_idx][self.name_col]

    def to_skycoords(self):
        """
        Returns a SkyCoord representation of the known objects.
        """
        return SkyCoord(ra=self.data[self.ra_col], dec=self.data[self.dec_col], unit="deg")

    def match(self, result_data, wcs):
        """This function takes a list of results and matches them to known objects.

        This modifies the `Results` table by adding a column with name `self.matcher_name` that provides for each
        result a dictionary mapping the names of known objects (as defined by the catalog's `name_col`) to a boolean
        array indicating which observations in the result matched to that known object. Note that depending on the
        matching parameters, a result can match to multiple known objects from the catalog even at the same observation time.

        So for a dataset with 5 observations a result matching to 2 known objects, A and B, might have an entry in
        the column `self.matcher_name` like:
        ```{
            "A": [True, True, False, False, False],
            "B": [False, False, False, True, True],
        }```

        Parameters
        ----------
        result_data: `Results`
            The set of results to filter. This data gets modified directly by
            the filtering.
        wcs: `astropy.wcs.WCS`
            The common WCS object for the stack of images.

        Returns
        -------
        `Results`
            The modified `Results` object returned for chaining.
        """
        logger.info(f"Matching known objects to {len(result_data)} results using {self.matcher_name} filter")
        all_matches = []

        # Get the RA and DEC of the known objects and the trajectories of the results for matching
        known_objs_ra_dec = self.to_skycoords()
        trj_list = result_data.make_trajectory_list()

        for result_idx in range(len(result_data)):
            # Generate (RA, Dec) pairs for all of the valid observations for this result trajectory
            valid_obstimes = self.obstimes[result_data[result_idx]["obs_valid"]]
            trj_skycoords = trajectory_predict_skypos(trj_list[result_idx], wcs, valid_obstimes)

            # Because we're only matching using the subset of obstimes that were valid for this result, we
            # can use this to later map back to the original index of all observations in the stack.
            trj_idx_to_obs_idx = np.where(result_data[result_idx]["obs_valid"])[0]

            # Now we can compare the SkyCoords of the known objects to the SkyCoords of the result trajectories
            # using search_around_sky.
            # This will return a list of indices of known objects that are within sep_thresh of a trajectory
            # Note that subsequent calls by default will use the same underlying KD-Tree iin coords2.cache.
            trjs_idx, known_objs_idx, _, _ = search_around_sky(
                trj_skycoords, known_objs_ra_dec, self.sep_thresh
            )

            # Now we can count per-known object how many observations matched within this result
            matched_known_objs = {}
            for t_idx, ko_idx in zip(trjs_idx, known_objs_idx):
                # The observation spatially matched but now check that the time separation is witihin our threshold
                if abs(self.get_mjd(ko_idx) - valid_obstimes[t_idx]) * 24 * 3600 <= self.time_thresh_s:
                    # The name of the object that matched to this observation
                    obj_name = self.get_name(ko_idx)
                    if obj_name not in matched_known_objs:
                        # Create an array of which observations match to this object.
                        # Note that we need to use the length of all obstimes, not just the presently valid ones
                        matched_known_objs[obj_name] = np.full(len(self.obstimes), False)
                    # Map to the original set of all obstimes (valid or invalid) since that's what we
                    # want for results filtering.
                    obs_idx = trj_idx_to_obs_idx[t_idx]
                    matched_known_objs[obj_name][obs_idx] = True
            all_matches.append(matched_known_objs)

        # Add matches as a result column
        result_data.table[self.matcher_name] = all_matches

        logger.info(f"Matched known objects to {len(result_data)} results using {self.matcher_name} filter")

        return result_data

    def mark_matched_obs_invalid(
        self,
        result_data,
        drop_empty_rows=True,
    ):
        """
        Mark observations that matched to known objects as invalid, by default dropping
        results that no longer have any valid observations.

        Note that a given result can match to multiple objects, and that we expect the
        `Results` table to have a column with name corresponding to `self.matcher_name` that
        contains which observations were matched to each known object.

        Parameters
        ----------
        result_data : `Results`
            The results to filter.
        drop_empty_rows : bool, optional
            If True, drop rows that have no valid observations after filtering. Default is True.

        Returns
        -------
        `Results`
            The modified `Results` object returned for chaining.
        """
        # Skip filtering if there is nothing to filter.
        if len(result_data) == 0 or len(self.obstimes) == 0 or len(self.data) == 0:
            return result_data

        if self.matcher_name not in result_data.table.colnames:
            raise ValueError(
                f"Column {self.matcher_name} not found in results table. Please run match() first."
            )

        matched_known_objs = result_data.table[self.matcher_name]
        new_obs_valid = result_data["obs_valid"]
        for result_idx in range(len(result_data)):
            # A result can match to multiple objects, so we want to logically OR
            # against all matching objects with a logical OR using np.any.
            # We can then use bitwise NOT and AND to mark any previously valid
            # observations that matched to known objects as invalid.
            new_obs_valid[result_idx] &= ~np.any(
                np.array(list(matched_known_objs[result_idx].values())), axis=0
            )

        return result_data.update_obs_valid(new_obs_valid, drop_empty_rows=drop_empty_rows)

    def match_on_min_obs(
        self,
        result_data,
        min_obs,
    ):
        """
        Create a column corresponding to the known objects that were matched to a result
        based on the minimum number of observations that matched to that known object.
        Note that the ratio is calculated based on the total number of observations
        that were within `time_sep_thresh_s` of the `obstimes` we are matching to. Observations
        outside of that time range are not considered.

        Note that a given result can match to multiple objects.

        Parameters
        ----------
        result_data : `Results`
            The results to filter.
        min_obs : int
            The minimum number of observations within a KBMOD result that must match to a known
            object for that result to be considered a match.

        Returns
        -------
        `Results`
            The modified `Results` object returned for chaining.
        """
        if self.matcher_name not in result_data.table.colnames:
            raise ValueError(
                f"Column {self.matcher_name} not found in results table. Please run match() first."
            )

        matched_objs = []
        for idx in range(len(result_data)):
            matched_objs.append(set([]))
            matches = result_data[self.matcher_name][idx]
            for name in matches:
                if np.count_nonzero(matches[name]) >= min_obs:
                    matched_objs[-1].add(name)
        result_data.table[self.match_min_obs_col(min_obs)] = matched_objs

        return result_data

    def match_on_obs_ratio(
        self,
        result_data,
        obs_ratio,
    ):
        """
        Create a column corresponding to the known objects that were matched to a result
        based on the proportion of observations that matched to that known object within the catalog.

        Note that a given result can match to multiple objects.

        Parameters
        ----------
        result_data : `Results`
            The results to filter.
        obs_ratio : float
            The minimum ratio of observations within a KBMOD result that must match to the total
            observations within our catalog of known objects for that result to be considered a match.
            Must be within the range [0, 1].

        Returns
        -------
        `Results`
            The modified `Results` object returned for chaining.

        Raises
        ------
        ValueError
            If `obs_ratio` is not within the range [0, 1].
        """
        if obs_ratio < 0 or obs_ratio > 1:
            raise ValueError("obs_ratio must be within the range [0, 1].")

        if self.matcher_name not in result_data.table.colnames:
            raise ValueError(
                f"Column {self.matcher_name} not found in results table. Please run match() first."
            )

        # Create a dictionary of how many observations we have for each known object
        # in our catalog
        known_obj_cnts = dict(Counter(self.data[self.name_col]))
        matched_objs = []
        for idx in range(len(result_data)):
            matched_objs.append(set([]))
            matches = result_data[self.matcher_name][idx]
            for name in matches:
                if name not in known_obj_cnts:
                    raise ValueError(f"Unknown known object {name}")

                curr_obs_ratio = np.count_nonzero(matches[name]) / known_obj_cnts[name]
                if curr_obs_ratio <= obs_ratio:
                    matched_objs[-1].add(name)

        result_data.table[self.match_obs_ratio_col(obs_ratio)] = matched_objs

        return result_data

    def get_recovered_objects(self, result_data, match_col):
        """
        Get the set of objects that were recovered or missed in the results.

        For our purposes, a recovered object is one that was matched to a result based on the
        matching column of choice in the results table and a missing object are objects in
        the catalog that were not matched. Note that not all catalogs may be
        constructed in a way where all objects could be spatially present and
        recoverable in the results.

        Parameters
        ----------
        result_data : `Results`
            The results to filter.
        match_col : str
            The name of the column in the results table that contains the matched objects.

        Returns
        -------
        set, set
            A tuple of sets where the first set contains the names of objects that were recovered
            and the second set contains the names objects that were missed

        Raises
        ------
        ValueError
            If the `match_col` is not present in the results table
        """
        if match_col not in result_data.table.colnames:
            raise ValueError(f"Column {match_col} not found in results table.")

        if len(result_data) == 0 or len(self.data) == 0:
            return set(), set()

        expected_objects = set(self.data[self.name_col])
        matched_objects = set()
        for idx in range(len(result_data)):
            matched_objects.update(result_data[match_col][idx])
        recovered_objects = matched_objects.intersection(expected_objects)
        missed_objects = expected_objects - recovered_objects

        return recovered_objects, missed_objects

    def filter_matches(self, result_data, match_col):
        """
        Filter out the results table to only include results that did not match to any known objects.

        Parameters
        ----------
        result_data : `Results`
            The results to filter.
        match_col : str
            The name of the column in the results table that contains the matched objects.

        Returns
        -------
        `Results`
            The modified `Results` object returned for chaining.

        Raises
        ------
        ValueError
            If the `match_col` is not present in the results table.
        """
        if match_col not in result_data.table.colnames:
            raise ValueError(f"Column {match_col} not found in results table.")

        if len(result_data) == 0:
            return result_data

        # Only keep results that did not match to any known objects in our column
        idx_to_keep = np.array([len(x) == 0 for x in result_data[match_col]])
        # Use the name of our matching column as the filter name
        result_data = result_data.filter_rows(idx_to_keep, match_col)

        return result_data
