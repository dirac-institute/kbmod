import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky

import kbmod.search as kb
from kbmod.trajectory_utils import trajectory_predict_skypos

logger = kb.Logging.getLogger(__name__)


class KnownObjsMatcher:
    """
    A class which ingests an astopy table of KnownObj data for easy maniuplation.
    """

    def __init__(
        self,
        table,
        obstimes,
        filter_params,
        mjd_col="mjd_mid",
        ra_col="RA",
        dec_col="DEC",
        name_col="Name",
    ):
        """
        Parameters
        ----------
        table : astropy.table.Table
            The table containing the known objects.
        mjd_col : str
            The name of the column containing the MJD of the known objects.
        ra_col : str
            The name of the column containing the RA of the known objects.
        dec_col : str
            The name of the column containing the DEC of the known objects.
        name_col : str
            The name of the column containing the name of the known objects.

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
        self.filter_params = filter_params
        if len(self.obstimes) == 0:
            raise ValueError("No obstimes provided")
        if not self.filter_params:
            raise ValueError("No filter_params provided")

        self.time_sep_thresh_s = (
            self.filter_params["known_obj_sep_time_thresh_s"]
            if "known_obj_sep_time_thresh_s" in self.filter_params
            else 1200.0
        )
        self.filter_rows_by_time()

        # Column names for recovered objects using either filtering method.
        self.match_min_obs_col = "recovered_" + self.filter_params["filter_type"] + "_min_obs"
        self.match_obs_ratio_col = "recovered_" + self.filter_params["filter_type"] + "_obs_ratio"

    def __len__(self):
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

    def filter_rows_by_time(self):
        """
        Returns a new KnownObjs filtered down to within a given time range.

        Parameters
        ----------
        start_mjd : float
            The start of the time range.
        end_mjd : float
            The end of the time range.

        Returns
        -------
        KnownObjs
            A new KnownObjs object filtered down withing the given time range.
        """
        start_mjd = max(0, min(self.obstimes) - 2 - self.time_sep_thresh_s)
        end_mjd = max(self.obstimes) + 2 + self.time_sep_thresh_s

        self.data = self.data[(self.data[self.mjd_col] >= start_mjd) & (self.data[self.mjd_col] <= end_mjd)]

    def get_counts(self):
        """
        Returns the number of known objects in the table.
        """
        cnts = {}
        for i in range(len(self.data)):
            if self.data[self.name_col][i] not in cnts:
                cnts[self.data[self.name_col][i]] = 0
            cnts[self.data[self.name_col][i]] += 1

        return cnts

    def match_known_obj_filters(self, result_data, wcs):
        """This function takes a list of results and matches them to known objects.

        Parameters
        ----------
        result_data: `Results`
            The set of results to filter. This data gets modified directly by
            the filtering.
        known_objs: `KnownObjs`
            The known objects to filter against.
        obstimes: list(float)
            The mjd times of each possible observation.
        wcs: `astropy.wcs.WCS`
            The common WCS object for the stack of images.
        filter_params : dict
            Contains values concerning the image and search settings including:
            filter_type, filter_params, and filter_obs.
        remove_match_obs : bool
            If True, remove observations that match to known objects from the results

        Returns
        -------
        list(dict)
            A list where each element is a dictionary of matching known objects for the
            corresponding result in ``result_data``. The dictionary maps the name of the
            known object to a boolean array of length equal to the number of valid observations
            in the result. Each element of the array is True if that known object matched to
            the corresponding observation, and False otherwise.

        Raises
        ------
            Raises a ValueError if the parameters are not valid.
            Raises a TypeError if ``result_data`` is of an unsupported type.
        """
        all_matches = []

        sep_thresh = (
            self.filter_params["known_obj_sep_thresh"]
            if "known_obj_sep_thresh" in self.filter_params
            else 1.0
        )
        sep_thresh = sep_thresh * u.arcsec

        # Get the RA and DEC of the known objects and the trajectories of the results for matching
        known_objs_ra_dec = self.to_skycoords()
        trj_list = result_data.make_trajectory_list()

        new_obs_valid = result_data["obs_valid"]
        for result_idx in range(len(result_data)):
            valid_obstimes = self.obstimes[result_data[result_idx]["obs_valid"]]
            trj_skycoords = trajectory_predict_skypos(trj_list[result_idx], wcs, valid_obstimes)
            # Becauase we're only using the valid obstimes, we can user this below to map back to
            # the original observation index.
            trj_idx_to_obs_idx = np.where(result_data[result_idx]["obs_valid"])[0]

            # Now we can compare the SkyCoords of the known objects to the SkyCoords of the trajectories using search_around_sky
            # This will return a list of indices of known objects that are within sep_thresh of a trajectory
            trjs_idx, known_objs_idx, _, _ = search_around_sky(trj_skycoords, known_objs_ra_dec, sep_thresh)

            # Now we can count per-known object how many observations matched within this result
            matched_known_objs = {}
            for t_idx, ko_idx in zip(trjs_idx, known_objs_idx):
                # Check the time separation is witihin our threshold
                if abs(self.get_mjd(ko_idx) - valid_obstimes[t_idx]) * 3600 <= self.time_sep_thresh_s:
                    # The name of the object that matched to this observation
                    obj_name = self.get_name(ko_idx)
                    # Create an array of dimension trj_skycoords where each value is false
                    if obj_name not in matched_known_objs:
                        # Note that we need to use the length of all obstimes, not just the presently valid ones
                        matched_known_objs[obj_name] = np.full(len(self.obstimes), False)
                    # Map to the original of all obstimes (valid or invalid) since that's what we
                    # want for results filtering.
                    obs_idx = trj_idx_to_obs_idx[t_idx]
                    if obs_idx >= len(matched_known_objs[obj_name]):
                        raise ValueError(
                            f"obs_idx: {obs_idx}, \n t_idx: {t_idx}, \n trj_idx_to_obs_idx: {trj_idx_to_obs_idx}, \nvalid_obstimes: {valid_obstimes}\n,trjs_idx: {trjs_idx},\n known_objs_idx: {known_objs_idx}"
                        )

                    matched_known_objs[obj_name][obs_idx] = True
            all_matches.append(matched_known_objs)
            # Only consider observations valid if they did not match to any known objects
            new_obs_valid[result_idx] &= ~np.any(np.array(list(matched_known_objs.values())), axis=0)

        # Add matches as a result column
        result_data.table[self.filter_params["filter_type"]] = all_matches

        return result_data

    def apply_known_obj_valid_obs_filter(
        self,
        result_data,
        wcs,
        update_obs_valid=True,
    ):
        # Skip filtering if there is nothing to filter.
        if len(result_data) == 0 or len(self.obstimes) == 0:
            logger.info(f'{self.filter_params["filter_type"]} : skipping, no results.')
            return []

        # Skip matching known objects if there are none
        if len(self.data) == 0:
            logger.info("Known Object Filtering : skipping, no objects to match agains.")
            return []
        result_data = self.match_known_obj_filters(result_data, wcs)

        matched_known_objs = result_data.table[self.filter_params["filter_type"]]
        if update_obs_valid:
            new_obs_valid = result_data["obs_valid"]
            for result_idx in range(len(result_data)):
                # A result can match to multiple objects, so we want to AND our valid
                # obsesrvations against against all known objects that matched.
                new_obs_valid[result_idx] &= ~np.any(
                    np.array(list(matched_known_objs[result_idx].values())), axis=0
                )
            result_data.update_obs_valid(new_obs_valid)

        return matched_known_objs

    def apply_known_obj_match_min_obs(
        self,
        result_data,
    ):
        matched_objs = []
        for idx in range(len(result_data)):
            matched_objs.append(set([]))
            matches = result_data[self.filter_params["filter_type"]][idx]
            for name in matches:
                if np.count_nonzero(matches[name]) >= self.filter_params["known_obj_match_min_obs"]:
                    matched_objs[-1].add(name)
        result_data.table[self.match_min_obs_col] = matched_objs
        return result_data

    def apply_known_obj_match_obs_ratio(
        self,
        result_data,
    ):
        known_obj_cnts = self.get_counts()

        matched_objs = []
        for idx in range(len(result_data)):
            matched_objs.append(set([]))
            matches = result_data[self.filter_params["filter_type"]][idx]
            for name in matches:
                if name not in known_obj_cnts:
                    raise ValueError(f"Unknown known object {name}")

                obs_ratio = np.count_nonzero(matches[name]) / known_obj_cnts[name]
                if obs_ratio <= self.filter_params["known_obj_match_obs_ratio"]:
                    matched_objs[-1].add(name)

        result_data.table[self.match_obs_ratio_col] = matched_objs
        return result_data

    def filter_known_obj(
        self,
        result_data,
        match_col,
        filter_label,
    ):
        # Only keep results that did not match to any known objects in our specified column
        idx_to_keep = np.array([len(x) == 0 for x in result_data[match_col]])
        return result_data.filter_rows(idx_to_keep, filter_label)

    def get_recovered_objects(self, result_data, expected_objects, match_obj_col):
        """ """
        matched_objects = set(result_data[match_obj_col])
        recovered_objects = matched_objects.intersection(expected_objects)
        missing_objects = expected_objects - recovered_objects

        return recovered_objects, missing_objects
