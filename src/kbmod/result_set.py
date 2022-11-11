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


class ResultDataRow:
    """
    This class stores a collection of related data from a single kbmod result.
    """
    __slots__ = ("_trajectory",
                 "_stamp",
                 "_final_likelihood",
                 "_valid_times",
                 "_valid_indices",
                 "_all_stamps",
                 "_psi_curve",
                 "_phi_curve",
                 "_num_times",
                )
    
    def __init__(self, trj, times):
        self._trajectory = trj
        self._stamp = None
        self._final_likelihood = trj.lh
        self._valid_times = copy.copy(times)
        self._valid_indices = [i for i in range(len(times))]
        self._all_stamps = None
        self._psi_curve = None
        self._phi_curve = None
        self._num_times = len(times)

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def stamp(self):
        return self._stamp

    @property
    def final_lh(self):
        return self._final_likelihood

    @property
    def final_likelihood(self):
        return self._final_likelihood

    @property
    def valid_times(self):
        return self._valid_times

    @property
    def valid_indices(self):
        return self._valid_indices

    @property
    def all_stamps(self):
        return self._all_stamps

    @property
    def psi_curve(self):
        return self._psi_curve

    @property
    def phi_curve(self):
        return self._phi_curve

    def set_stamp(self, stamp):
        self._stamp = stamp

    def set_all_stamps(self, all_stamps):
        self._all_stamps = all_stamps

    def set_psi_phi(self, psi, phi):
        """
        Set the psi and phi curves and auto calculate the light curve.
        
        Arguments:
            psi : List - The psi curve
            phi : List - The phi curve
        """
        assert(len(psi) == len(phi))
        self._psi_curve = psi
        self._phi_curve = phi
        self._update_likelihood()

    def get_trj_result(self):
        """
        Return the current trajectory information as a trj_result object.
        """
        return kb.trj_result(self._trajectory, self._num_times, self._valid_indices)

    def filter_indices(self, indices_to_keep):
        """
        Remove invalid indices and times from the ResultDataRow. This uses relative filtering
        where valid_indices[i] is kept for all i in indices_to_keep.
        
        Arguments:
            indices_to_keep : List of ints - which indices to keep.
        """
        self._valid_indices = [self._valid_indices[i] for i in indices_to_keep]
        self._valid_times = [self._valid_times[i] for i in indices_to_keep]
        self._update_likelihood()

    def get_filtered_psi(self):
        """
        Return a list of psi values from the valid indices. Used for doing
        repeat filtering or debugging.
        """
        assert(self._psi_curve is not None)
        return [self._psi_curve[i] for i in self._valid_indices]

    def get_filtered_phi(self):
        """
        Return a list of phi values from the valid indices. Used for doing
        repeat filtering or debugging.
        """
        assert(self._phi_curve is not None)
        return [self._phi_curve[i] for i in self._valid_indices]

    def compute_light_curve(self, only_valid_indices=False):
        """
        Fill the light curve from the psi and phi curves.
        """
        if self._psi_curve is None or self._phi_curve is None:
            return []

        num_elements = len(self._psi_curve)
        assert(num_elements == len(self._phi_curve))
        lc = [0.0] * num_elements
        for i in range(num_elements):
            if self._phi_curve[i] != 0.0:
                lc[i] = self._psi_curve[i] / self._phi_curve[i]
        return lc
    
    def compute_likelihood_curve(self):
        """
        Compute the likelihood curve for each point (based on psi and phi).
        """
        assert(self._psi_curve is not None)
        assert(self._phi_curve is not None)

        num_elements = len(self._psi_curve)
        assert(num_elements == len(self._phi_curve))
        lh = [0.0] * num_elements
        for i in range(num_elements):
            if self._phi_curve[i] > 0.0:
                lh[i] = self._psi_curve[i] / math.sqrt(self._phi_curve[i])
        return lh

    def _update_likelihood(self):
        if self.psi_curve is None or self.phi_curve is None:
            return

        psi_sum = 0.0
        phi_sum = 0.0
        for ind in self._valid_indices:
            psi_sum += self._psi_curve[ind]
            phi_sum += self._phi_curve[ind]

        if phi_sum <= 0.0:
            self._final_likelihood = 0.0
        else:
            self._final_likelihood = psi_sum / math.sqrt(phi_sum)
            
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

    def trajectory_list(self, skip_if_none=False, indices_to_use=None):
        """
        Create and return a list of just the trajectories.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
            indices_to_use : list
                A list of indices to ouput. Use None to return all trajectories.
        """
        arr = []
        if indices_to_use is None:
            arr = [x.trajectory for x in self.results]
        else:
            arr = [self.results[i].trajectory for i in indices_to_use]

        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def trj_result_list(self, skip_if_none=False, indices_to_use=None):
        """
        Create and return a list of just the trajectory result objects.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
            indices_to_use : list
                A list of indices to ouput. Use None to return all.
        """
        arr = []
        if indices_to_use is None:
            arr = [x.get_trj_result() for x in self.results]
        else:
            arr = [self.results[i].get_trj_result() for i in indices_to_use]

        if skip_if_none and any(v is None for v in arr):
            return []
        return arr
    
    def final_likelihood_list(self, skip_if_none=False):
        """
        Create and return a list of just the final likelihoods.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.final_lh for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def valid_times_list(self, skip_if_none=False):
        """
        Create and return a list of just the valid times arrays.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.valid_times for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def valid_indices_list(self, skip_if_none=False):
        """
        Create and return a list of just the valid indices arrays.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.valid_indices for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def light_curve_list(self, skip_if_none=False):
        """
        Create and return a list of just the light curves.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.compute_light_curve() for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def psi_curve_list(self, skip_if_none=False):
        """
        Create and return a list of just the psi curves.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.psi_curve for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def phi_curve_list(self, skip_if_none=False):
        """
        Create and return a list of just the phi curves.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.phi_curve for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def stamp_list(self, skip_if_none=False):
        """
        Create and return a list of just the stamps.
        
        Arguments:
            skip_if_none : bool
                Output an empty array if ANY of the elements are None.
        """
        arr = [x.stamp for x in self.results]
        if skip_if_none and any(v is None for v in arr):
            return []
        return arr

    def all_stamps_list(self, skip_if_none=False):
        """
        Create and return a list of just the all_stamps lists.
        
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
        keep["results"] = self.trajectory_list(True)
        keep["new_lh"] = self.final_likelihood_list(True)
        keep["times"] = self.valid_times_list(True)
        keep["lc"] = self.light_curve_list(True)
        keep["lc_index"] = self.valid_indices_list(True)
        keep["psi_curves"] = self.psi_curve_list(True)
        keep["phi_curves"] = self.phi_curve_list(True)
        keep["stamps"] = self.stamp_list(True)
        keep["all_stamps"] = self.all_stamps_list(True)
        keep["final_results"] = [i for i in range(len(self.results))]
        
        return keep

    def append_result_dict(self, res_dict, all_times):
        """
        Append all the results in a dictionary (as defined by gen_results_dict)
        to the current result set. Used for backwards compatibility.

        WARNING: This function writes to the "private" fileds of ResultDataRow
        to ensure full backwards compatibility.
        
        Arguments:
            res_dict : dictionary of results
            all_times : a list of all the times
        """
        inds_to_use = []
        if np.any(res_dict["final_results"] == ...):
            inds_to_use = [i for i in range(len(res_dict["results"]))]
        else:
            inds_to_use = res_dict["final_results"]

        for i in inds_to_use:
            row = ResultDataRow(res_dict["results"][i], all_times)
            if len(res_dict["new_lh"]) > i:
                row._final_likelihood = res_dict["new_lh"][i]
            if len(res_dict["times"]) > i:
                row._valid_times = res_dict["times"][i]
            if len(res_dict["lc_index"]) > i:
                row._valid_indices = res_dict["lc_index"][i]
            if len(res_dict["psi_curves"]) > i:
                row._psi_curve = res_dict["psi_curves"][i]
            if len(res_dict["phi_curves"]) > i:
                row._phi_curve = res_dict["phi_curves"][i]
            self.results.append(row)

        # The 'stamps' and 'all_stamps' entries are treated oddly by the legacy code and
        # not indexed by final_results. Instead the stamps are prefiltered to match
        # final_results. We need to copy them over separately.
        num_results = len(inds_to_use)
        if (len(res_dict["stamps"]) > 0):
            assert(len(res_dict["stamps"]) == num_results)
            for i in range(num_results):
                self.results[i]._stamp = res_dict["stamps"][i]
        if (len(res_dict["all_stamps"]) > 0):
            assert(len(res_dict["all_stamps"]) == num_results)
            for i in range(num_results):
                self.results[i]._all_stamps = res_dict["all_stamps"][i]
                                                           
    def filter_results(self, indices_to_keep):
        """
        Filter the rows in the ResultSet to only include those indices
        in the list indices_to_keep.
        
        Arguments:
            indices_to_keep : List of int
        """
        self.results = [self.results[i] for i in indices_to_keep]

    def filter_on_num_valid_indices(self, min_valid_indices):
        """
        Filter out rows with fewer than min_valid_indices valid indices.
        
        Arguments:
            min_valid_indices : int
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
            threshold : float
        """
        tmp_results = []
        for x in self.results:
            if x.final_likelihood >= threshold:
                tmp_results.append(x)
        self.results = tmp_results

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
            np.array(self.trajectory_list(True)),     
            fmt="%s"
        ) 
        with open("%s/lc_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.light_curve_list(True))
        with open("%s/psi_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.psi_curve_list(True))
        with open("%s/phi_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.phi_curve_list(True))
        with open("%s/lc_index_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.valid_indices_list(True))
        with open("%s/times_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.valid_times_list(True))
        np.savetxt(
            "%s/filtered_likes_%s.txt" % (res_filepath, out_suffix),
            np.array(self.final_likelihood_list(True)),
            fmt="%.4f"
        )
        stamps_list = np.array(self.stamp_list(True))
        np.savetxt(
            "%s/ps_%s.txt" % (res_filepath, out_suffix),
            stamps_list.reshape(len(stamps_list), 441),
            fmt="%.4f"
        )
        stamps_to_save = np.array(self.all_stamps_list(True))
        np.save("%s/all_ps_%s.npy" % (res_filepath, out_suffix), stamps_to_save)
