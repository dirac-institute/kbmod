import math
import multiprocessing as mp
import os.path as ospath
import numpy as np

from kbmod.file_utils import *


class ResultRow:
    """This class stores a collection of related data from a single kbmod result."""

    __slots__ = (
        "trajectory",
        "stamp",
        "all_stamps",
        "final_likelihood",
        "valid_indices",
        "psi_curve",
        "phi_curve",
        "num_times",
    )

    def __init__(self, trj, num_times):
        self.trajectory = trj
        self.stamp = None
        self.final_likelihood = trj.lh
        self.valid_indices = [i for i in range(num_times)]
        self.all_stamps = None
        self.psi_curve = None
        self.phi_curve = None
        self.num_times = num_times

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
        return [all_times[i] for i in self.valid_indices]

    @property
    def light_curve(self):
        """Compute the light curve from the psi and phi curves.

        Returns
        -------
        lc : list
            The likelihood curve. This is an empty list if either
            psi or phi are not set.
        """
        if self.psi_curve is None or self.phi_curve is None:
            return []

        num_elements = len(self.psi_curve)
        lc = [0.0] * num_elements
        for i in range(num_elements):
            if self.phi_curve[i] != 0.0:
                lc[i] = self.psi_curve[i] / self.phi_curve[i]
        return lc

    @property
    def likelihood_curve(self):
        """Compute the likelihood curve for each point (based on psi and phi).

        Returns
        -------
        lh : list
            The likelihood curve. This is an empty list if either
            psi or phi are not set.
        """
        if self.psi_curve is None:
            raise ValueError("Psi curve is None")
        if self.phi_curve is None:
            raise ValueError("Phi curve is None")

        num_elements = len(self.psi_curve)
        lh = [0.0] * num_elements
        for i in range(num_elements):
            if self.phi_curve[i] > 0.0:
                lh[i] = self.psi_curve[i] / math.sqrt(self.phi_curve[i])
        return lh

    def valid_indices_as_booleans(self):
        """Get a Boolean vector indicating which indices are valid.

        Returns
        -------
        result : list
            A list of bool indicating which indices appear in valid_indices
        """
        indices_set = set(self.valid_indices)
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
        self.psi_curve = psi
        self.phi_curve = phi
        self._update_likelihood()

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
        current_num_inds = len(self.valid_indices)
        if any(v >= current_num_inds or v < 0 for v in indices_to_keep):
            raise ValueError(f"Out of bounds index in {indices_to_keep}")

        self.valid_indices = [self.valid_indices[i] for i in indices_to_keep]
        self._update_likelihood()

        # Update the count of valid observations in the trajectory object.
        self.trajectory.obs_count = len(self.valid_indices)

    def _update_likelihood(self):
        """Update the likelihood based on the result's psi and phi curves
        and the list of current valid indices.

        Note
        ----
        Requires that psi_curve and phi_curve have both been set. Otherwise
        defaults to a likelihood of 0.0.
        """
        if self.psi_curve is None or self.phi_curve is None:
            self.final_likelihood = 0.0
            return

        psi_sum = 0.0
        phi_sum = 0.0
        for ind in self.valid_indices:
            psi_sum += self.psi_curve[ind]
            phi_sum += self.phi_curve[ind]

        if phi_sum <= 0.0:
            self.final_likelihood = 0.0
            self.trajectory.lh = 0.0
            self.trajectory.flux = 0.0
        else:
            self.final_likelihood = psi_sum / math.sqrt(phi_sum)
            self.trajectory.lh = psi_sum / math.sqrt(phi_sum)
            self.trajectory.flux = psi_sum / phi_sum


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
        self.all_times = all_times
        self.results = []

        # Set up information to track which row is filtered at which round.
        self.track_filtered = track_filtered
        self.filtered = {}

    def num_results(self):
        """Return the number of results in the list.

        Returns
        -------
        int
            The number of results in the list.
        """
        return len(self.results)

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

    def zip_phi_psi_idx(self):
        """Create and return a list of tuples for each psi/phi curve.

        Returns
        -------
        iterable
            A list of tuples with (psi_curve, phi_curve, index) for
            each result in the ResultList.
        """
        return ((x.psi_curve, x.phi_curve, i) for i, x in enumerate(self.results))

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
        filter_obj : Filter
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
            [x.valid_times(self.all_times) for x in self.results],
            True,
        )
        FileUtils.save_csv_from_list(
            ospath.join(res_filepath, f"all_times_{out_suffix}.txt"),
            [self.all_times],
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
