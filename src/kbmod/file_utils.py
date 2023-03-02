from collections import OrderedDict
import csv

import numpy as np

import kbmod.search as kb


class FileUtils:
    def __init__(self):
        pass

    @staticmethod
    def load_time_dictionary(time_file):
        """Load a OrderedDict mapping visit_id to time stamp.

        Parameters
        ----------
        time_file : str
            The path and name of the time file.

        Returns
        -------
        image_time_dict : OrderedDict
            A mapping of visit ID to time stamp.
        """
        # Load a mapping from visit numbers to the visit times. This dictionary stays
        # empty if no time file is specified.
        image_time_dict = OrderedDict()
        if time_file is None or len(time_file) == 0:
            return image_time_dict

        with open(time_file, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=" ")
            for row in reader:
                if len(row[0]) < 2 or row[0][0] == "#":
                    continue
                image_time_dict[row[0]] = float(row[1])
        return image_time_dict

    @staticmethod
    def load_psf_dictionary(psf_file):
        """Load a OrderedDict mapping visit_id to PSF.

        Parameters
        ----------
        psf_file : str
            The path and name of the PSF file.

        Returns
        -------
        psf_dict : OrderedDict
            A mapping of visit ID to psf value.
        """
        # Load a mapping from visit numbers to the visit times. This dictionary stays
        # empty if no time file is specified.
        psf_dict = OrderedDict()
        if psf_file is None or len(psf_file) == 0:
            return psf_dict

        with open(psf_file, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=" ")
            for row in reader:
                if len(row[0]) < 2 or row[0][0] == "#":
                    continue
                psf_dict[row[0]] = float(row[1])
        return psf_dict

    @staticmethod
    def trajectory_from_np_object(result):
        """Transform a numpy object holding trajectory information
        into a trajectory object.

        Parameters
        ----------
        result : np object
            The result object loaded by numpy.

        Returns
        -------
        trj : trajectory
            The corresponding trajectory object.
        """
        trj = kb.trajectory()
        trj.x = int(result["x"])
        trj.y = int(result["y"])
        trj.x_v = float(result["vx"])
        trj.y_v = float(result["vy"])
        trj.flux = float(result["flux"])
        trj.lh = float(result["lh"])
        trj.obs_count = int(result["num_obs"])
        return trj

    @staticmethod
    def load_results_file(filename):
        """Load the result trajectories.

        Parameters
        ----------
        filename : str
            The filename of the results.

        Returns
        -------
        results : np array
            A np array with the result trajectories.
        """
        results = np.genfromtxt(
            filename,
            usecols=(1, 3, 5, 7, 9, 11, 13),
            names=["lh", "flux", "x", "y", "vx", "vy", "num_obs"],
        )
        return results

    @staticmethod
    def load_results_file_as_trajectories(filename):
        """Load the result trajectories.

        Parameters
        ----------
        filename : str
            The full path and filename of the results.

        Returns
        -------
        results : list
            A list of trajectory objects
        """
        np_results = FileUtils.load_results_file(filename)
        results = [FileUtils.trajectory_from_np_object(x) for x in np_results]
        return results
