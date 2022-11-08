import copy
import csv
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


class ResultDataRow:
    """
    This class stores a collection of related data from a single kbmod result.
    """
    __slots__ = ("trajectory",
                 "stamp",
                 "final_lh",
                 "lc",
                 "valid_times",
                 "valid_indices",
                 "all_stamps",
                 "psi_curve",
                 "phi_curve",
                 "num_times",
                )
    
    def __init__(self, trj, times):
        self.trajectory = trj
        self.stamp = None
        self.final_lh = trj.lh
        self.lc = None
        self.valid_times = copy.copy(times)
        self.valid_indices = [i for i in range(len(times))]
        self.all_stamps = None
        self.psi_curve = None
        self.phi_curve = None
        self.num_times = len(times)

    def get_trj_result(self):
        """
        Return the current trajectory information as a trj_result object.
        """
        return kb.trj_result(self.trajectory, self.num_times, self.valid_indices)
    
    def filter_indices(self, indices_to_keep, filter_curves=False, filter_stamps=False):
        """
        Remove invalid indices and times from the ResultDataRow. This uses relative filtering
        where valid_indices[i] is kept for all i in indices_to_keep.
        
        Arguments:
            indices_to_keep : List of ints - which indices to keep.
            filter_curves : bool - indicates whether to filter the curve arrays
            filter_stamps : bool - indicates whether to filter the stamp arrays
        """
        self.valid_indices = [self.valid_indices[i] for i in indices_to_keep]
        self.valid_times = [self.valid_times[i] for i in indices_to_keep]
        if filter_curves:
            if self.psi_curve is not None:
                self.psi_curve = [self.psi_curve[i] for i in indices_to_keep]
            if self.phi_curve is not None:
                self.phi_curve = [self.phi_curve[i] for i in indices_to_keep]
            if self.lc is not None:
                self.lc = [self.lc[i] for i in indices_to_keep]
        if filter_stamps and self.all_stamps is not None:
            self.all_stamps = [self.all_stamps[i] for i in indices_to_keep]

    def fill_lc_from_psi_phi(self):
        """
        Fill the light curve from the psi and phi curves.
        """
        if self.psi_curve is None or self.phi_curve is None:
            self.lc = None
            return

        num_elements = len(self.psi_curve)
        assert(num_elements == len(self.phi_curve))
        self.lc = [0.0] * num_elements
        for i in range(num_elements):
            if self.phi_curve[i] != 0.0:
                self.lc[i] = self.psi_curve[i] / self.phi_curve[i]

class ResultSet:
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
        Add a single ResultDataRow to the result set.

        Arguments:
            res : ResultDataRow
        """
        self.results.append(res)

    def append_result_list(self, result_list):
        """
        Append multiple results from a list.

        Arguments:
            result_list : A list of ResultDataRows.
        """
        for x in result_list:
            self.results.append(x)

    def append_result_set(self, result_set):
        """
        Append the results in a second ResultsSet to the current one.

        Arguments:
            result_set : ResultsSet
        """
        for x in result_set.results:
            self.results.append(x)

    def trajectory_array(self, skip_if_none=False):
        """
        Create and return an array of just the trajectories.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.trajectory for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def final_lh_array(self, skip_if_none=False):
        """
        Create and return an array of just the final likelihoods.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.final_lh for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def valid_times_array(self, skip_if_none=False):
        """
        Create and return an array of just the valid times arrays.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.valid_times for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def valid_indices_array(self, skip_if_none=False):
        """
        Create and return an array of just the valid indices arrays.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.valid_indices for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def lc_array(self, skip_if_none=False):
        """
        Create and return an array of just the light curves.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.lc for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def psi_curve_array(self, skip_if_none=False):
        """
        Create and return an array of just the psi curves.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.psi_curve for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def phi_curve_array(self, skip_if_none=False):
        """
        Create and return an array of just the phi curves.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.phi_curve for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def stamp_array(self, skip_if_none=False):
        """
        Create and return an array of just the stamps.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.stamp for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def all_stamps_array(self, skip_if_none=False):
        """
        Create and return an array of just the all_stamps lists.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.all_stamps for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr
        
    def to_result_dict(self):
        """
        Transform the ResultsSet into a dictionary as defined by gen_results_dict.
        Used for backwards compatibility.
        """
        st = SharedTools()
        keep = st.gen_results_dict()
        keep["results"] = self.trajectory_array(True)
        keep["new_lh"] = self.final_lh_array(True)
        keep["times"] = self.valid_times_array(True)
        keep["lc"] = self.lc_array(True)
        keep["lc_index"] = self.valid_indices_array(True)
        keep["psi_curves"] = self.psi_curve_array(True)
        keep["phi_curves"] = self.phi_curve_array(True)
        keep["stamps"] = self.stamp_array(True)
        keep["all_stamps"] = self.all_stamps_array(True)
        keep["final_results"] = [i for i in range(len(self.results))]
        
        return keep

    def append_result_dict(self, res_dict):
        """
        Append all the results in a dictionary (as defined by gen_results_dict)
        to the current result set. Used for backwards compatibility.
        
        Arguments:
            res_dict : dictionary of results
        """
        inds_to_use = []
        if np.any(res_dict["final_results"] == ...):
            inds_to_use = [i for i in range(len(res_dict["results"]))]
        else:
            inds_to_use = res_dict["final_results"]

        for i in inds_to_use:
            row = ResultDataRow(res_dict["results"][i], [])
            if len(res_dict["new_lh"]) > i:
                row.final_lh = res_dict["new_lh"][i]
            if len(res_dict["times"]) > i:
                row.valid_times = res_dict["times"][i]
            if len(res_dict["lc"]) > i:
                row.lc = res_dict["lc"][i]
            if len(res_dict["lc_index"]) > i:
                row.valid_indices = res_dict["lc_index"][i]
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
            assert(len(res_dict["stamps"]) == num_results)
            for i in range(num_results):
                self.results[i].stamp = res_dict["stamps"][i]
        if (len(res_dict["all_stamps"]) > 0):
            assert(len(res_dict["all_stamps"]) == num_results)
            for i in range(num_results):
                self.results[i].all_stamps = res_dict["all_stamps"][i]
                                                           
    def filter_results(self, indices_to_keep):
        """
        Filter the rows in the ResultSet to only include those indices
        in the list indices_to_keep.
        
        Arguments:
            indices_to_keep : List of int
        """
        self.results = [self.results[i] for i in indices_to_keep]
        
    def save_to_files(self, res_filepath, out_suffix):
        """
        This function saves results from a search method to a series of files.
        INPUT-
            res_filepath : string
            out_suffix : string
                Suffix to append to the output file name
        """
        np.savetxt(
            "%s/results_%s.txt" % (res_filepath, out_suffix),
            np.array(self.trajectory_array(True)),     
            fmt="%s"
        ) 
        with open("%s/lc_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.lc_array(True))
        with open("%s/psi_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.psi_curve_array(True))
        with open("%s/phi_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.phi_curve_array(True))
        with open("%s/lc_index_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.valid_indices_array(True))
        with open("%s/times_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.valid_times_array(True))
        np.savetxt(
            "%s/filtered_likes_%s.txt" % (res_filepath, out_suffix),
            np.array(self.final_lh_array(True)),
            fmt="%.4f"
        )
        stamps_list = np.array(self.stamp_array(True))
        np.savetxt(
            "%s/ps_%s.txt" % (res_filepath, out_suffix),
            stamps_list.reshape(len(stamps_list), 441),
            fmt="%.4f"
        )
        stamps_to_save = np.array(self.all_stamps_array(True))
        np.save("%s/all_ps_%s.npy" % (res_filepath, out_suffix), stamps_to_save)
