import copy
import csv
import math
import numpy as np
import kbmod.search as kb

class SharedTools:
    """
    This class manages tools that are shared by the classes Interface and
    PostProcess.
    
    Legacy approach. Soon to be deprecated.
    """

    def __init__(self):
        return

    def gen_results_dict(self):
        """
        Return an empty results dictionary. All values needed for a results
        dictionary should be added here. This dictionary gets passed into and
        out of most Interface and PostProcess methods, getting altered and/or
        replaced by filtering along the way.
        """
        keep = {
            "stamps": [],
            "new_lh": [],
            "results": [],
            "times": [],
            "lc": [],
            "lc_index": [],
            "all_stamps": [],
            "psi_curves": [],
            "phi_curves": [],
            "final_results": ...,
        }
        return keep


class ResultRow:
    """
    This class stores a collection of related data from a single kbmod result.
    """
    __slots__ = ("trajectory",
                 "stamp",
                 "all_stamps",
                 "final_likelihood",
                 "all_times",
                 "valid_indices",
                 "psi_curve",
                 "phi_curve",
                 "num_times",
                )
    
    def __init__(self, trj, times):
        self.trajectory = trj
        self.stamp = None
        self.final_likelihood = trj.lh
        self.all_times = copy.copy(times)
        self.valid_indices = [i for i in range(len(times))]
        self.all_stamps = None
        self.psi_curve = None
        self.phi_curve = None
        self.num_times = len(times)

    @property
    def valid_times(self):
        """
        Get the times for the indices marked as valid.

        Returns:
            list: The times for the valid indices.
        """
        return [self.all_times[i] for i in self.valid_indices]

    @property
    def trj_result(self):
        """
        Get the current trajectory information as a trj_result object.

        Returns:
            trj_result: The trajectory information.
        """
        return kb.trj_result(self.trajectory, self.num_times, self.valid_indices)

    @property
    def light_curve(self):
        """
        Compute the light curve from the psi and phi curves.

        Returns:
            list: The likelihood curve. This is an empty list if either
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
        """
        Compute the likelihood curve for each point (based on psi and phi).

        Returns:
            list: The likelihood curve. This is an empty list if either
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

    def set_psi_phi(self, psi, phi):
        """
        Set the psi and phi curves and auto calculate the light curve.
        
        Arguments:
            psi (list): The psi curve
            phi (list): The phi curve

        Raises:
            ValueError: If the phi and psi lists are not the same length
                as the number of times.
        """
        if len(psi) != len(phi) or len(psi) != self.num_times:
            raise ValueError(f"Expected arrays of length {self.num_times} got {len(phi)} and {len(psi)} instead")
        self.psi_curve = psi
        self.phi_curve = phi
        self._update_likelihood()

    def filter_indices(self, indices_to_keep):
        """
        Remove invalid indices and times from the ResultRow. This uses relative filtering
        where valid_indices[i] is kept for all i in indices_to_keep. Updates the
        trajectory's likelihood using only the new valid indices.

        Arguments:
            indices_to_keep : List of ints - which indices to keep.

        Raises:
            ValueError: If any of the given indices are out of bounds.
        """
        current_num_inds = len(self.valid_indices)
        if any(v >= current_num_inds or v < 0 for v in indices_to_keep):
            raise ValueError(f"Out of bounds index in {indices_to_keep}")

        self.valid_indices = [self.valid_indices[i] for i in indices_to_keep]
        self._update_likelihood()
        
    def _update_likelihood(self):
        """
        Update the likelihood based on the result's psi and phi curves
        and the list of current valid indices.

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
        else:
            self.final_likelihood = psi_sum / math.sqrt(phi_sum)


class ResultList:
    """
    This class stores a collection of related data from all of the kbmod results.
    """
    def __init__(self):
        self.results = []

    def num_results(self):
        """
        Return the number of results in the list.
        """
        return len(self.results)
    
    def clear(self):
        """
        Clear the list of results.
        """
        self.results.clear()

    def append_result(self, res):
        """
        Add a single ResultRow to the result set.

        Arguments:
            res (ResultRow): The new result to add.
        """
        self.results.append(res)

    def extend(self, result_list):
        """
        Append the results in a second ResultSet to the current one.

        Arguments:
            result_list (ResultList): The set of new ResultRow
        """
        self.results.extend(result_list.results)

    def trajectory_list(self, indices_to_use=None):
        """
        Create and return a list of just the trajectories.

        Arguments:
            indices_to_use (list): A list of indices to ouput.
                Use None to return all trajectories.
        """
        if indices_to_use is not None:
            return [self.results[i].trajectory for i in indices_to_use]
        return [x.trajectory for x in self.results]

    def trj_result_list(self, indices_to_use=None):
        """
        Create and return a list of just the trajectory result objects.

        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
            indices_to_use : list
                A list of indices to ouput. Use None to return all.
        """
        if indices_to_use is not None:
            return [self.results[i].get_trj_result() for i in indices_to_use]
        return [x.get_trj_result() for x in self.results]
        
    def to_result_dict(self):
        """
        Transform the ResultsSet into a dictionary as defined by gen_results_dict.
        Used for backwards compatibility.

        Returns:
            dict: A results dictionary as generated by SharedTools.gen_results_dict.
        """
        st = SharedTools()
        keep = st.gen_results_dict()
        for row in self.results:
            keep["results"].append(row.trajectory)
            keep["new_lh"].append(row.final_likelihood)
            keep["times"].append(row.valid_times)
            keep["lc"].append(row.light_curve)
            keep["lc_index"].append(row.valid_indices)
            keep["psi_curves"].append(row.psi_curve)
            keep["phi_curves"].append(row.phi_curve)

        # Generate the stamps and keep the array only if none of them are None.
        keep["stamps"] = [x.stamp for x in self.results]
        if any(v is None for v in keep["stamps"]):
            keep["stamps"] = []

        # Generate all_stamps and keep the array only if none of them are None.
        keep["all_stamps"] = [x.all_stamps for x in self.results]
        if any(v is None for v in keep["all_stamps"]):
            keep["all_stamps"] = []

        # The final results are every row from the ResultList.
        keep["final_results"] = [i for i in range(len(self.results))]

        return keep

    def append_result_dict(self, res_dict, all_times):
        """
        Append all the results in a dictionary (as defined by gen_results_dict)
        to the current result set. Used for backwards compatibility.

        Arguments:
            res_dict (dict): dictionary of results
            all_times (list): a list of all the times
        """
        inds_to_use = []
        if np.any(res_dict["final_results"] == ...):
            inds_to_use = [i for i in range(len(res_dict["results"]))]
        else:
            inds_to_use = res_dict["final_results"]

        for i in inds_to_use:
            row = ResultRow(res_dict["results"][i], all_times)
            if len(res_dict["new_lh"]) > i:
                row.final_likelihood = res_dict["new_lh"][i]
            if len(res_dict["lc_index"]) > i:
                row.valid_indices = res_dict["lc_index"][i]
            if len(res_dict["times"]) > i:
                # Overwrite the "all_times" with the corresponding entry
                # accounting for the fact that the results dictionary
                # only keeps times corresponding to the valid indices.
                for new_ind, old_ind in enumerate(row.valid_indices):
                    row.all_times[old_ind] = res_dict["times"][i][new_ind]
            if len(res_dict["psi_curves"]) > i:
                row.psi_curve = res_dict["psi_curves"][i]
            if len(res_dict["phi_curves"]) > i:
                row.phi_curve = res_dict["phi_curves"][i]
            self.results.append(row)

        # The 'stamps' and 'all_stamps' entries are treated oddly by the legacy code and
        # not indexed by final_results. Instead the stamps are prefiltered to match
        # final_results. We need to copy them over separately.
        num_results = len(inds_to_use)
        if (len(res_dict["stamps"]) > 0):
            for i in range(num_results):
                self.results[i].stamp = res_dict["stamps"][i]
        if (len(res_dict["all_stamps"]) > 0):
            for i in range(num_results):
                self.results[i].all_stamps = res_dict["all_stamps"][i]
                                                           
    def filter_results(self, indices_to_keep):
        """
        Filter the rows in the ResultSet to only include those indices
        in the list indices_to_keep.
        
        Arguments:
            indices_to_keep (list): The indices of the rows to keep.
        """
        self.results = [self.results[i] for i in indices_to_keep]

    def filter_on_num_valid_indices(self, min_valid_indices):
        """
        Filter out rows with fewer than min_valid_indices valid indices.
        
        Arguments:
            min_valid_indices (int): The minimum number of valid indices
                a row needs to pass the filtering.
        """
        tmp_results = []
        for x in self.results:
            if len(x.valid_indices) >= min_valid_indices:
                tmp_results.append(x)
        self.results = tmp_results

    def filter_on_likelihood(self, threshold):
        """
        Filter out rows with a final likelihood below the threshold.
        
        Arguments:
            threshold (float): The minimum likelihood a row needs to
                pass the filtering.
        """
        tmp_results = []
        for x in self.results:
            if x.final_likelihood >= threshold:
                tmp_results.append(x)
        self.results = tmp_results

    def save_to_files(self, res_filepath, out_suffix):
        """
        This function saves results from a search method to a series of files.

        Arguments:
            res_filepath (string): The directory in which to store the results.
            out_suffix (string): The suffix to append to the output file name
        """
        np.savetxt(
            "%s/results_%s.txt" % (res_filepath, out_suffix),
            np.array([x.trajectory for x in self.results]),     
            fmt="%s"
        ) 
        with open("%s/lc_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows([x.light_curve for x in self.results])
        with open("%s/psi_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows([x.psi_curve for x in self.results])
        with open("%s/phi_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows([x.phi_curve for x in self.results])
        with open("%s/lc_index_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows([x.valid_indices for x in self.results])
        with open("%s/times_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows([x.valid_times for x in self.results])
        np.savetxt(
            "%s/filtered_likes_%s.txt" % (res_filepath, out_suffix),
            np.array([x.final_likelihood for x in self.results]),
            fmt="%.4f"
        )

        stamps_list = np.array([x.stamp for x in self.results])
        if np.any(stamps_list is None):
            stamps_list = np.array([])
        np.savetxt(
            "%s/ps_%s.txt" % (res_filepath, out_suffix),
            stamps_list.reshape(len(stamps_list), 441),
            fmt="%.4f"
        )

        # Save the "all stamps" file.
        stamps_to_save = np.array([x.all_stamps for x in self.results])
        if np.any(stamps_to_save is None):
            stamps_to_save = np.array([])
        np.save("%s/all_ps_%s.npy" % (res_filepath, out_suffix), stamps_to_save)
