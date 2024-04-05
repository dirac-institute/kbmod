"""ResultList is a row-based data structure for tracking results with additional logic for
filtering and maintaining consistency between different attributes in each row. Each row is
represented as a ResultRow.
"""

import math
import multiprocessing as mp
import numpy as np
import os.path as ospath
from pathlib import Path

from astropy.table import Table
from yaml import dump, safe_load

from kbmod.file_utils import *
from kbmod.trajectory_utils import (
    make_trajectory,
    trajectory_from_yaml,
    trajectory_predict_skypos,
    trajectory_to_yaml,
)


def _check_optional_allclose(arr1, arr2):
    """Check whether toward optional numpy arrays have the same information.

    Parameters
    ----------
    arr1 : `numpy.ndarray` or `None`
        The first array.
    arr1 : `numpy.ndarray` or `None`
        The second array.

    Returns
    -------
    result : `bool`
        Indicates whether the arrays are the same.
    """
    if arr1 is None and arr2 is None:
        return True
    if arr1 is not None and arr2 is None:
        return False
    if arr1 is None and arr2 is not None:
        return False
    return np.allclose(arr1, arr2)


class ResultRow:
    """This class stores a collection of related data from a single kbmod result.
    In order to maintain a consistent internal state, the class uses private variables
    with getter only properties for key pieces of information. While this adds overhead,
    it requires the users to work through specific setter functions that update one
    attribute in relation to another. Do not set any of the private data members directly.

    Example:
    To set the psi and phi curves use: set_psi_phi(psi, phi). This will update the curves
    the final likelihood, the trajectory's likelihood, and the trajectory's flux.

    Most attributes are optional so we only use space for the ones needed. The attributes
    are set to None when unused.

    Attributes
    ----------
    all_stamps : `numpy.ndarray`
        An array of numpy arrays representing stamps for each timestep
        (including the invalid/filtered ones). [Optional]
    final_likelihood : `float`
        The final likelihood as computed from the valid indices of the psi and phi
        curves. Initially set from the trajectory's lh field. [Required]
    num_times : `int`
        The number of timesteps. [Required]
    stamp : `numpy.ndarray`
        The coadded stamp computed from the valid timesteps. [Optional]
    phi_curve : `list` or `numpy.ndarray`
        An array of numpy arrays representing phi values for each timestep
        (including the invalid/filtered ones). [Optional]
    pred_dec : `numpy.ndarray`
        An array of the predict positions dec. [Optional]
    pred_ra : `numpy.ndarray`
        An array of the predict positions RA. [Optional]
    psi_curve : `list` or `numpy.ndarray`
        An array of numpy arrays representing psi values for each timestep
        (including the invalid/filtered ones). [Optional]
    trajectory : `kbmod.search.Trajectory`
        The result trajectory in pixel space. [Required]
    valid_indices : `list` or `numpy.ndarray`
        The indices of the timesteps that are unfiltered (valid). [Required]
    """

    __slots__ = (
        "all_stamps",
        "_final_likelihood",
        "_num_times",
        "_phi_curve",
        "pred_ra",
        "pred_dec",
        "_psi_curve",
        "stamp",
        "trajectory",
        "_valid_indices",
    )

    def __init__(self, trj, num_times):
        self.all_stamps = None
        self._final_likelihood = trj.lh
        self._num_times = num_times
        self._phi_curve = None
        self.pred_dec = None
        self.pred_ra = None
        self._psi_curve = None
        self.stamp = None
        self.trajectory = trj
        self._valid_indices = [i for i in range(num_times)]

    @classmethod
    def from_table_row(cls, data, num_times=None):
        """Create a ResultRow object directly from an AstroPy Table row.

        Parameters
        ----------
        data : 'astropy.table.row.Row'
            The incoming row.
        all_times : `int`, optional
            The number of total times in the data. If ``None`` tries
            to extract from a "num_times" or "all_times" column.

        Raises
        ------
        KeyError if a column is missing.
        """
        if num_times is None:
            if "num_times" in data.columns:
                num_times = data["num_times"]
            elif "all_times" in data.columns:
                num_times = len(data["all_times"])
            else:
                raise KeyError("Number of times is not specified.")

        # Create the Trajectory object from the correct fields.
        trj = make_trajectory(
            data["trajectory_x"],
            data["trajectory_y"],
            data["trajectory_vx"],
            data["trajectory_vy"],
            data["flux"],
            data["likelihood"],
            data["obs_count"],
        )

        # Manually fill in all the rest of the values. We let the stamp related columns
        # be empty to save space.
        row = ResultRow(trj, num_times)
        row._final_likelihood = data["likelihood"]
        row._phi_curve = data["phi_curve"]
        row.pred_dec = data["pred_dec"]
        row.pred_ra = data["pred_ra"]
        row._psi_curve = data["psi_curve"]
        row._valid_indices = data["valid_indices"]

        if "all_stamps" in data.columns:
            row.all_stamps = data["all_stamps"]
        else:
            row.all_stamps = None

        if "stamp" in data.columns:
            row.stamp = data["stamp"]
        else:
            row.stamp = None

        return row

    def __eq__(self, other):
        """Test if two result rows are equal."""
        if not isinstance(other, ResultRow):
            return False

        # Check the attributes of the trajectory first.
        if (
            self.trajectory.x != other.trajectory.x
            or self.trajectory.y != other.trajectory.y
            or self.trajectory.vx != other.trajectory.vx
            or self.trajectory.vy != other.trajectory.vy
            or self.trajectory.lh != other.trajectory.lh
            or self.trajectory.flux != other.trajectory.flux
            or self.trajectory.obs_count != other.trajectory.obs_count
        ):
            return False

        # Check the simple attributes.
        if not self._num_times == other._num_times:
            return False
        if not self._final_likelihood == other._final_likelihood:
            return False

        # Check the curves and stamps.
        if not _check_optional_allclose(self.all_stamps, other.all_stamps):
            return False
        if not _check_optional_allclose(self._phi_curve, other._phi_curve):
            return False
        if not _check_optional_allclose(self._psi_curve, other._psi_curve):
            return False
        if not _check_optional_allclose(self.stamp, other.stamp):
            return False
        if not _check_optional_allclose(self._valid_indices, other._valid_indices):
            return False
        if not _check_optional_allclose(self.pred_dec, other.pred_dec):
            return False
        if not _check_optional_allclose(self.pred_ra, other.pred_ra):
            return False

        return True

    @property
    def final_likelihood(self):
        return self._final_likelihood

    @property
    def valid_indices(self):
        return self._valid_indices

    @property
    def psi_curve(self):
        return self._psi_curve

    @property
    def phi_curve(self):
        return self._phi_curve

    @property
    def num_times(self):
        return self._num_times

    @property
    def obs_count(self):
        return self.trajectory.obs_count

    @property
    def flux(self):
        return self.trajectory.flux

    @classmethod
    def from_yaml(cls, yaml_str):
        """Deserialize a ResultRow from a YAML formatted string.

        Parameters
        ----------
        yaml_str : `str`
            The YAML string to deserialize.
        """
        yaml_params = safe_load(yaml_str)

        # Access the minimum values to create the ResultRow object.
        trj = trajectory_from_yaml(yaml_params["trajectory"])
        num_times = yaml_params["num_times"]
        result = ResultRow(trj, num_times)

        # Copy the values into the object.
        for attr in ResultRow.__slots__:
            if attr != "trajectory":
                attr_name = attr.lstrip("_")
                setattr(result, attr, yaml_params[attr_name])

        # Convert the stamps to np arrays
        if result.stamp is not None:
            result.stamp = np.array(result.stamp)
        if result.all_stamps is not None:
            result.all_stamps = np.array(result.all_stamps)

        return result

    def to_yaml(self):
        """Serialize a ResultRow from a YAML formatted string.

        Parameters
        ----------
        yaml_str : `str`
            The YAML string to deserialize.
        """
        yaml_dict = {"trajectory": trajectory_to_yaml(self.trajectory)}

        for attr in ResultRow.__slots__:
            if attr != "trajectory":
                value = getattr(self, attr)

                # Strip numpy types which cannot be safely loaded by YAML.
                if type(value) is np.ndarray:
                    value = value.tolist()
                elif type(value) is np.float64:
                    value = float(value)

                attr_name = attr.lstrip("_")
                yaml_dict[attr_name] = value
        return dump(yaml_dict)

    def valid_times(self, all_times):
        """Get the times for the indices marked as valid.

        Parameters
        ----------
        all_times : list
            A list of all time stamps.

        Returns
        -------
        list
            The times for the valid indices.
        """
        return [all_times[i] for i in self._valid_indices]

    @property
    def light_curve(self):
        """Compute the light curve from the psi and phi curves.

        Returns
        -------
        lc : `numpy.ndarray`
            The light curve. This is an empty array if either
            psi or phi are not set.
        """
        if self._psi_curve is None or self._phi_curve is None:
            return np.array([])

        masked_phi = np.copy(self._phi_curve)
        masked_phi[masked_phi == 0] = 1e12
        lc = np.divide(self._psi_curve, masked_phi)
        return lc

    @property
    def likelihood_curve(self):
        """Compute the likelihood curve for each point (based on psi and phi).

        Returns
        -------
        lh : `numpy.ndarray`
            The likelihood curve. This is an empty array if either
            psi or phi are not set.
        """
        if self._psi_curve is None or self._phi_curve is None:
            return np.array([])

        masked_phi = np.copy(self._phi_curve)
        masked_phi[masked_phi == 0] = 1e12
        lh = np.divide(self._psi_curve, np.sqrt(masked_phi))
        return lh

    def valid_indices_as_booleans(self):
        """Get a Boolean vector indicating which indices are valid.

        Returns
        -------
        result : list
            A list of bool indicating which indices appear in valid_indices
        """
        indices_set = set(self._valid_indices)
        result = [(x in indices_set) for x in range(self.num_times)]
        return result

    def set_psi_phi(self, psi, phi):
        """Set the psi and phi curves and auto calculate the light curve.

        Parameters
        ----------
        psi : list
            The psi curve.
        phi : list
            The phi curve.

        Raises
        ------
        ValueError
            If the phi and psi lists are not the same length as the number
            of times.
        """
        if len(psi) != len(phi) or len(psi) != self.num_times:
            raise ValueError(
                f"Expected arrays of length {self.num_times} got {len(phi)} and {len(psi)} instead"
            )
        self._psi_curve = psi
        self._phi_curve = phi
        self._update_likelihood()

    def compute_predicted_skypos(self, times, wcs):
        """Set the predicted sky positions at each time.

        Parameters
        ----------
        times : `list` or `numpy.ndarray`
            The times at which to predict the positions.
        wcs : `astropy.wcs.WCS`
            The WCS for the images.
        """
        if len(times) != self.num_times:
            raise ValueError(f"Expected an array of length {self.num_times} got {len(times)} instead")
        sky_pos = trajectory_predict_skypos(self.trajectory, wcs, times)

        self.pred_ra = sky_pos.ra.value
        self.pred_dec = sky_pos.dec.value

    def filter_indices(self, indices_to_keep):
        """Remove invalid indices and times from the ResultRow. This uses relative
        filtering where valid_indices[i] is kept for all i in indices_to_keep.
        Updates the trajectory's likelihood using only the new valid indices.

        Parameters
        ----------
        indices_to_keep : list
            A list of which of the current indices to keep.

        Raises
        ------
        ValueError: If any of the given indices are out of bounds.
        """
        current_num_inds = len(self._valid_indices)
        if any(v >= current_num_inds or v < 0 for v in indices_to_keep):
            raise ValueError(f"Out of bounds index in {indices_to_keep}")

        self._valid_indices = [self._valid_indices[i] for i in indices_to_keep]
        self._update_likelihood()

        # Update the count of valid observations in the trajectory object.
        self.trajectory.obs_count = len(self._valid_indices)

    def _update_likelihood(self):
        """Update the likelihood based on the result's psi and phi curves
        and the list of current valid indices.

        Note
        ----
        Requires that psi_curve and phi_curve have both been set. Otherwise
        does not perform any updates.
        """
        if self._psi_curve is None or self._phi_curve is None:
            return

        psi_sum = 0.0
        phi_sum = 0.0
        for ind in self._valid_indices:
            psi_sum += self._psi_curve[ind]
            phi_sum += self._phi_curve[ind]

        if phi_sum <= 0.0:
            self._final_likelihood = 0.0
            self.trajectory.lh = 0.0
            self.trajectory.flux = 0.0
        else:
            self._final_likelihood = psi_sum / np.sqrt(phi_sum)
            self.trajectory.lh = psi_sum / np.sqrt(phi_sum)
            self.trajectory.flux = psi_sum / phi_sum

    def append_to_dict(self, result_dict, expand_trajectory=False):
        """A helper function for transforming a ResultList into a dictionary.
        Appends the row's data onto the various lists in a dictionary. Users should
        note need to call this directly.

        Parameter
        ---------
        result_dict : `dict`
            The dictionary to extend.
        expand_trajectory : `bool`
            Expand each entry in trajectory into its own entry.
        """
        if expand_trajectory:
            result_dict["trajectory_x"].append(self.trajectory.x)
            result_dict["trajectory_y"].append(self.trajectory.y)
            result_dict["trajectory_vx"].append(self.trajectory.vx)
            result_dict["trajectory_vy"].append(self.trajectory.vy)
            result_dict["obs_count"].append(self.trajectory.obs_count)
            result_dict["flux"].append(self.trajectory.flux)
        else:
            result_dict["trajectory"].append(trajectory)
        result_dict["likelihood"].append(self._final_likelihood)

        result_dict["stamp"].append(self.stamp)
        result_dict["all_stamps"].append(self.all_stamps)
        result_dict["valid_indices"].append(self._valid_indices)
        result_dict["psi_curve"].append(self._psi_curve)
        result_dict["phi_curve"].append(self._phi_curve)
        result_dict["pred_ra"].append(self.pred_ra)
        result_dict["pred_dec"].append(self.pred_dec)


class ResultList:
    """This class stores a collection of related data from all of the kbmod results."""

    def __init__(self, all_times, track_filtered=False):
        """Create a ResultList class.

        Parameters
        ----------
        all_times : list
            A list of all time stamps.
        track_filtered : bool
            Whether to track (save) the filtered trajectories. This will use
            more memory and is recommended only for analysis.
        """
        self._all_times = all_times
        self.results = []

        # Set up information to track which row is filtered at which round.
        self.track_filtered = track_filtered
        self.filtered = {}

    # All times should be externally read-only once set.
    @property
    def all_times(self):
        return self._all_times

    @classmethod
    def from_yaml(cls, yaml_str):
        """Deserialize a ResultList from a YAML string.

        Parameters
        ----------
        yaml_str : `str`
            The serialized string.
        """
        yaml_dict = safe_load(yaml_str)
        result_list = ResultList(yaml_dict["all_times"], yaml_dict["track_filtered"])
        result_list.results = [ResultRow.from_yaml(row) for row in yaml_dict["results"]]

        if result_list.track_filtered:
            for key in yaml_dict["filtered"]:
                result_list.filtered[key] = [ResultRow.from_yaml(row) for row in yaml_dict["filtered"][key]]
        return result_list

    @classmethod
    def from_table(self, data, all_times=None, track_filtered=False):
        """Extract the ResultList from an astropy Table.

        Parameters
        ----------
        data : `astropy.table.Table`
            The input data.
        all_times : `List` or `numpy.ndarray` or None
            The list of all time stamps. Must either be set or there
            must be an all_times column in the Table.
        track_filtered : `bool`
            Indicates whether the ResultList should track future filtered points.

        Raises
        ------
        KeyError if any columns are missing or if is ``all_times`` is None and there
        is no all_times column in the data.
        """
        # Check that we have some list of time stamps and place it in all_times.
        if all_times is None:
            if "all_times" not in data.columns:
                raise KeyError(f"No time stamps provided.")
            else:
                all_times = data["all_times"][0]
        num_times = len(all_times)

        result_list = ResultList(all_times, track_filtered)
        for i in range(len(data)):
            row = ResultRow.from_table_row(data[i], num_times)
            result_list.append_result(row)
        return result_list

    @classmethod
    def read_table(self, filename):
        """Read the ResultList from a table file.

        Parameters
        ----------
        filename : `str`
            The name of the file to load.


        Raises
        ------
        FileNotFoundError if the file is not found.
        KeyError if any of the columns are missing.
        """
        if not Path(filename).is_file():
            raise FileNotFoundError
        data = Table.read(filename)
        return ResultList.from_table(data)

    def num_results(self):
        """Return the number of results in the list.

        Returns
        -------
        int
            The number of results in the list.
        """
        return len(self.results)

    def __len__(self):
        """Return the number of results in the list."""
        return len(self.results)

    def __eq__(self, other):
        """Test if two ResultLists are equal. Includes both ordering and values."""
        if not isinstance(other, ResultList):
            return False
        if not np.allclose(self._all_times, other._all_times):
            return False
        if self.track_filtered != other.track_filtered:
            return False

        num_results = len(self.results)
        if num_results != len(other.results):
            return False
        for i in range(num_results):
            if self.results[i] != other.results[i]:
                return False

        if len(self.filtered) != len(other.filtered):
            return False
        for key in self.filtered.keys():
            if key not in other.filtered:
                return False

            num_filtered = len(self.filtered[key])
            if num_filtered != len(other.filtered):
                return False
            for i in range(num_filtered):
                if self.filtered[key][i] != other.filtered[key][i]:
                    return False

        return True

    def clear(self):
        """Clear the list of results."""
        self.results.clear()
        self.filtered.clear()

    def append_result(self, res):
        """Add a single ResultRow to the result set.

        Parameters
        ----------
        res : `ResultRow`
            The new result to add.
        """
        self.results.append(res)

    def extend(self, result_list):
        """Append the results in a second ResultSet to the current one.

        Parameters
        ----------
        result_list : `ResultList`
            The data structure containing additional `ResultRow` elements to add.
        """
        self.results.extend(result_list.results)

        # When merging the filtered results extend lists with the
        # same key and create new lists for new keys.
        for key in result_list.filtered.keys():
            if key in self.filtered:
                self.filtered[key].extend(result_list.filtered[key])
            else:
                self.filtered[key] = result_list.filtered[key]

    def sort(self, key="final_likelihood", reverse=True):
        """Sort the results by the given key. This must correspond
        to one of the proporties in ResultRow.

        Parameters
        ----------
        key : `str`
            A string representing the property by which to sort.
            Default = final_likelihood
        reverse : `bool`
            Sort in increasing order.

        Returns
        -------
        self : ResultList
            Returns a reference to itself to allow chaining.
        """
        self.results.sort(key=lambda x: getattr(x, key), reverse=reverse)
        return self

    def get_result_values(self, attribute):
        """Return the values of the ResultRows' attribute as a list.
        Subattributes can be extracted as "attribute.subattribute",
        such as "trajectory.x"

        Examples:
            trj_list = res.get_trajectory_values("trajectory")
            x_values = res.get_trajectory_values("trajectory.x")
            stamps = res.get_trajectory_values("stamp")

        Parameter
        ---------
        attribute : `str`
            The name of the attribute to extract.

        Returns
        -------
        values : `list`
            A list of the results' values.

        Raises
        ------
        Raises an ``AttributeError`` if the attribute does not exist.
        """
        att_list = attribute.split(".")

        values = []
        for row in self.results:
            obj = row
            for att in att_list:
                obj = getattr(obj, att)
            values.append(obj)
        return values

    def compute_predicted_skypos(self, wcs):
        """Compute the predict sky position for each result's trajectory
        at each time step.

        Parameters
        ----------
        wcs : `astropy.wcs.WCS`
            The WCS for the images.

        Returns
        -------
        self : ResultList
            Returns a reference to itself to allow chaining.
        """
        for row in self.results:
            row.compute_predicted_skypos(self._all_times, wcs)
        return self

    def filter_results(self, indices_to_keep, label=None):
        """Filter the rows in the ResultList to only include those indices
        in the list indices_to_keep.

        Parameters
        ----------
        indices_to_keep : list
            The indices of the rows to keep.
        label : string
            The label of the filtering stage to use. Only used if
            we keep filtered trajectories.

        Returns
        -------
        self : ResultList
            Returns a reference to itself to allow chaining.
        """
        if not self.track_filtered:
            # Deduplicate the indices to keep.
            dedup_inds = list(set(indices_to_keep))
            dedup_inds.sort()

            self.results = [self.results[i] for i in dedup_inds]
        else:
            keep_set = set(indices_to_keep)

            # Divide the current results into a set of rows to keep
            # and a set to filter.
            num_res = self.num_results()
            keep_rows = [self.results[i] for i in range(num_res) if i in keep_set]
            filter_rows = [self.results[i] for i in range(num_res) if i not in keep_set]

            # Set the main result list to be the kept rows.
            self.results = keep_rows

            # Add the filtered rows to the corresponding filter stage.
            if label is None:
                label = ""
            if label not in self.filtered:
                self.filtered[label] = []
            self.filtered[label].extend(filter_rows)

        # Return a reference to the current object to allow chaining.
        return self

    def apply_filter(self, filter_obj, num_threads=1):
        """Apply the given filter object to the ResultList.

        Modifies the ResultList in place.

        Parameters
        ----------
        filter_obj : RowFilter
            The filtering object to use.

        Returns
        -------
        self : ResultList
            Returns a reference to itself to allow chaining.
        """
        indices_to_keep = []
        if num_threads == 1:
            indices_to_keep = [i for i in range(self.num_results()) if filter_obj.keep_row(self.results[i])]
        else:
            pool = mp.Pool(processes=num_threads)
            keep_idx_results = pool.map_async(filter_obj.keep_row, self.results)
            pool.close()
            pool.join()
            keep_idx_results = keep_idx_results.get()
            indices_to_keep = [i for i in range(self.num_results()) if keep_idx_results[i]]
        self.filter_results(indices_to_keep, filter_obj.get_filter_name())

        return self

    def apply_batch_filter(self, filter_obj):
        """Apply the given batch filter object to the ResultList.

        Modifies the ResultList in place.

        Parameters
        ----------
        filter_obj : BatchFilter
            The filtering object to use.

        Returns
        -------
        self : ResultList
            Returns a reference to itself to allow chaining.
        """
        indices_to_keep = filter_obj.keep_indices(self)
        self.filter_results(indices_to_keep, filter_obj.get_filter_name())
        return self

    def get_filtered(self, label=None):
        """Get the results filtered at a given stage or all stages.

        Parameters
        ----------
        label : str
            The filtering stage to use. If no label is provided,
            return all filtered rows.

        Returns
        -------
        results : list
            A list of all filtered rows.
        """
        if not self.track_filtered:
            raise ValueError("ResultList filter tracking not enabled.")

        result = []
        if label is not None:
            # Check if anything was filtered at this stage.
            if label in self.filtered:
                result = self.filtered[label]
        else:
            for arr in self.filtered.values():
                result.extend(arr)

        return result

    def revert_filter(self, label=None):
        """Revert the filtering by re-adding filtered ResultRows.

        Note
        ----
        Filtered rows are appended to the end of the list. Does not return
        the results to the original ordering.

        Parameters
        ----------
        label : str
            The filtering stage to use. If no label is provided,
            revert all filtered rows.

        Returns
        -------
        self : ResultList
            Returns a reference to itself to allow chaining.

        Raises
        ------
        ValueError if filtering is not enabled.
        KeyError if label is unknown.
        """
        if not self.track_filtered:
            raise ValueError("ResultList filter tracking not enabled.")

        if label is not None:
            # Check if anything was filtered at this stage.
            if label in self.filtered:
                self.results.extend(self.filtered[label])
                del self.filtered[label]
            else:
                raise KeyError(f"Unknown filtered label {label}")
        else:
            for key in self.filtered:
                self.results.extend(self.filtered[key])

            # Reset the entire dictionary.
            self.filtered = {}

        return self

    def sync_table_indices(self, table):
        """Syncs the entries in the current list with those in a table version
        of the results by filtering on the 'index' column. Rows that do not
        appear in the table are removed from the list. The indices in the table
        are updated to match the new ordering in the result list.

        Parameters
        ----------
        table : `astropy.table.Table`
            A table with the data as generated by to_table().
        """
        self.filter_results(table["index"], "Table filtered")
        table["index"] = range(len(table))

    def to_table(self, filtered_label=None, append_times=False):
        """Extract the results into an astropy table.

        Parameters
        ----------
        filtered_label : `str`, optional
            The filtering label to extract. If None then extracts
            the unfiltered rows. (default=None)
        append_times : `bool`
            Append the list of all times as a column in the data.

        Returns
        -------
        table : `astropy.table.Table`
            A table with the data.

        Raises
        ------
        KeyError is the filtered_label is provided by does not exist.
        """
        # Choose the correct list to transform.
        if filtered_label is None:
            list_ref = self.results
        elif filtered_label in self.filtered:
            list_ref = self.filtered[filtered_label]
        else:
            raise KeyError(f"Unknown filter label {filtered_label}")

        table_dict = {
            "trajectory_x": [],
            "trajectory_y": [],
            "trajectory_vx": [],
            "trajectory_vy": [],
            "obs_count": [],
            "flux": [],
            "likelihood": [],
            "stamp": [],
            "valid_indices": [],
            "psi_curve": [],
            "phi_curve": [],
            "all_stamps": [],
            "pred_ra": [],
            "pred_dec": [],
        }
        if append_times:
            table_dict["all_times"] = []

        # Use a (slow) linear scan to do the transformation.
        for row in list_ref:
            row.append_to_dict(table_dict, True)
            if append_times:
                table_dict["all_times"].append(self._all_times)

        # Append the index information
        table_dict["index"] = [i for i in range(len(list_ref))]

        return Table(table_dict)

    def write_table(self, filename, overwrite=True, keep_all_stamps=False):
        """Write the unfiltered results to a single (ecsv) file.

        Parameter
        ---------
        filename : `str`
            The name of the result file.
        overwrite : `bool`
            Overwrite the file if it already exists. [default: True]
        keep_all_stamps : `bool`
            Keep individual stamps for each result and time step.
            This is very expensive and may fail if there are too many stamps.
            [default: False]
        """
        table_version = self.to_table(append_times=True)

        if not keep_all_stamps:
            table_version.remove_column("all_stamps")

        table_version.write(filename, overwrite=True)

    def to_yaml(self, serialize_filtered=False):
        """Serialize the ResultList as a YAML string.

        Parameters
        ----------
        serialize_filtered : `bool`
            Indicates whether or not to serialize the filtered results.

        Returns
        -------
        yaml_str : `str`
            The serialized string.
        """
        yaml_dict = {
            "all_times": self._all_times,
            "results": [row.to_yaml() for row in self.results],
            "track_filtered": False,
            "filtered": {},
        }

        if serialize_filtered and self.track_filtered:
            yaml_dict["track_filtered"] = True
            for key in self.filtered:
                yaml_dict["filtered"][key] = [row.to_yaml() for row in self.filtered[key]]

        return dump(yaml_dict)

    def save_to_files(self, res_filepath, out_suffix):
        """This function saves results from a search method to a series of files.

        Parameters
        ----------
        res_filepath : string
            The directory in which to store the results.
        out_suffix : string
            The suffix to append to the output file name
        """
        FileUtils.save_results_file(
            ospath.join(res_filepath, f"results_{out_suffix}.txt"),
            np.array([x.trajectory for x in self.results]),
        )
        FileUtils.save_csv_from_list(
            ospath.join(res_filepath, f"lc_{out_suffix}.txt"),
            [x.light_curve for x in self.results],
            True,
        )
        FileUtils.save_csv_from_list(
            ospath.join(res_filepath, f"psi_{out_suffix}.txt"),
            [x.psi_curve for x in self.results],
            True,
        )
        FileUtils.save_csv_from_list(
            ospath.join(res_filepath, f"phi_{out_suffix}.txt"),
            [x.phi_curve for x in self.results],
            True,
        )
        FileUtils.save_csv_from_list(
            ospath.join(res_filepath, f"lc_index_{out_suffix}.txt"),
            [x.valid_indices for x in self.results],
            True,
        )
        FileUtils.save_csv_from_list(
            ospath.join(res_filepath, f"times_{out_suffix}.txt"),
            [x.valid_times(self._all_times) for x in self.results],
            True,
        )
        FileUtils.save_csv_from_list(
            ospath.join(res_filepath, f"all_times_{out_suffix}.txt"),
            [self._all_times],
            True,
        )
        np.savetxt(
            ospath.join(res_filepath, f"filtered_likes_{out_suffix}.txt"),
            np.array([x.final_likelihood for x in self.results]),
            fmt="%.4f",
        )

        # Output the co-added stamps.
        stamps_list = np.array([x.stamp for x in self.results])
        if np.any(stamps_list == None):
            stamps_list = np.array([])

        stamp_size = 441
        if len(stamps_list) > 0:
            stamp_size = stamps_list[0].size

        np.savetxt(
            ospath.join(res_filepath, f"ps_{out_suffix}.txt"),
            stamps_list.reshape(len(stamps_list), stamp_size),
            fmt="%.4f",
        )

        # Save the "all stamps" file.
        stamps_to_save = np.array([x.all_stamps for x in self.results])
        if np.any(stamps_to_save == None):
            stamps_to_save = np.array([])
        np.save(ospath.join(res_filepath, f"all_ps_{out_suffix}.npy"), stamps_to_save)

        # If the ResultList has been tracking the filtered results, output them.
        if self.track_filtered:
            for label in self.filtered:
                fname = FileUtils.make_safe_filename(label)
                FileUtils.save_results_file(
                    ospath.join(res_filepath, f"filtered_results_{fname}_{out_suffix}.txt"),
                    np.array([x.trajectory for x in self.filtered[label]]),
                )


def load_result_list_from_files(res_filepath, suffix, all_mjd=None):
    """Create a new ResultList from outputted files.

    Parameters
    ----------
    res_filepath : string
        The directory in which the results are stored.
    suffix : string
        The suffix appended to the output file names.
    all_mjd : list
        A list of all the MJD timestamps (optional).
        If not provided, the function loads from the all_times file.

    Returns
    -------
    results : ResultList
       The results stored in the given directory with the correct suffix.
    """
    # Load the list of all time stamps unless they were pre-specified.
    if all_mjd is not None:
        all_times = all_mjd
    else:
        times_file_name = ospath.join(res_filepath, f"all_times_{suffix}.txt")
        all_times = FileUtils.load_csv_to_list(times_file_name, use_dtype=float)[0].tolist()
    num_times = len(all_times)

    # Create the ResultList to store the data.
    results = ResultList(all_times)

    # Load the one required file (trajectories).
    trajectories = FileUtils.load_results_file_as_trajectories(
        ospath.join(res_filepath, f"results_{suffix}.txt")
    )

    # Treat the remaining files as optional. Note that the lightcurve (lc_) can be computed
    # from psi + phi and times (time_) can be computed from lc_indices, so we do not need
    # to load those.
    psi = FileUtils.load_csv_to_list(
        ospath.join(res_filepath, f"psi_{suffix}.txt"), use_dtype=float, none_if_missing=True
    )
    phi = FileUtils.load_csv_to_list(
        ospath.join(res_filepath, f"phi_{suffix}.txt"), use_dtype=float, none_if_missing=True
    )
    lc_indices = FileUtils.load_csv_to_list(
        ospath.join(res_filepath, f"lc_index_{suffix}.txt"), use_dtype=int, none_if_missing=True
    )

    # Load the stamps file if it exists
    stamps = np.array([])
    stamps_file = ospath.join(res_filepath, f"ps_{suffix}.txt")
    if Path(stamps_file).is_file():
        stamps = np.genfromtxt(stamps_file)

        # If there is only one stamp, then numpy will load it as a 1-d array.
        # Change it to be 2-d.
        if len(np.shape(stamps)) < 2:
            stamps = np.array([stamps])
    num_stamps = len(stamps)

    # Load the all_stamps file if it exists.
    all_stamps = np.array([])
    all_stamps_file = ospath.join(res_filepath, f"all_ps_{suffix}.npy")
    if Path(all_stamps_file).is_file():
        all_stamps = np.load(all_stamps_file)
    num_all_stamps = len(all_stamps)

    for i in range(len(trajectories)):
        row = ResultRow(trajectories[i], num_times)

        # Handle the optional data
        if psi is not None and len(psi) > 0 and phi is not None and len(phi) > 0:
            row.set_psi_phi(psi[i], phi[i])
        if lc_indices is not None:
            row.filter_indices(lc_indices[i])
        if i < num_stamps and len(stamps[i]) > 0:
            row.stamp = stamps[i]
        if i < num_all_stamps and len(all_stamps[i]) > 0:
            row.all_stamps = all_stamps[i]

        # Append the result to the data set.
        results.append_result(row)

    return results
