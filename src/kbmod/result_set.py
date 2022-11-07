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
    def __init__(self, trj, times):
        self.trajectory = trj
        self.stamp = []
        self.final_lh = trj.lh
        self.lc = trj.lh
        self.valid_times = copy.copy(times)
        self.valid_indices = [i for i in range(len(times))]
        self.all_stamps = []
        self.psi_curve = []
        self.phi_curve = []

class ResultSet:
    """
    This class stores a collection of related data from all of the kbmod results.
    """
    def __init__(self):
        self.results = []

    def num_results(self):
        return len(self.results)

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

    def to_result_dict(self):
        """
        Transform the ResultsSet into a dictionary as defined by gen_results_dict.
        Used for backwards compatibility.
        """
        st = SharedTools()
        keep = st.gen_results_dict()
        for x in self.results:
            keep["results"].append(x.trajectory)
            keep["new_lh"].append(x.final_lh)
            keep["times"].append(x.valid_times)
            keep["lc"].append(x.lc)
            keep["lc_index"].append(x.valid_indices)
            keep["psi_curves"].append(x.psi_curve)
            keep["phi_curves"].append(x.phi_curve)

            # We treat stamps different during output.
            if x.stamp is not None and len(x.stamp) > 0:
                keep["stamps"].append(x.stamp)
            if x.all_stamps is not None and len(x.all_stamps) > 0:
                keep["all_stamps"].append(x.all_stamps)

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

    def trajectory_array(self):
        """
        Create and return an array of just the trajectories.
        """
        return [x.trajectory for x in self.results]
        
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
            np.array([x.trajectory for x in self.results]),     
            fmt="%s"
        )
        with open("%s/lc_%s.txt" % (res_filepath, out_suffix), "w") as f:
            writer = csv.writer(f)
            writer.writerows([x.lc for x in self.results])
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
            np.array([x.final_lh for x in self.results]),
            fmt="%.4f"
        )
        stamps_list = np.array([x.stamp for x in self.results])
        np.savetxt(
            "%s/ps_%s.txt" % (res_filepath, out_suffix),
            stamps_list.reshape(len(stamps_list), 441),
            fmt="%.4f"
        )
        stamps_to_save = np.array([x.all_stamps for x in self.results])
        np.save("%s/all_ps_%s.npy" % (res_filepath, out_suffix), stamps_to_save)
