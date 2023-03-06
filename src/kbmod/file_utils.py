"""A collection of utility functions for working with the input and
output files of KBMOD.
"""

from collections import OrderedDict
import csv
import numpy as np
from pathlib import Path

import kbmod.search as kb


class FileUtils:
    """A class of static methods for working with KBMOD files.

    Some examples
    * Load an external file of visit ID to timestamp mappings:
        ``time_dict = FileUtils.load_time_dictionary("kbmod/data/demo_times.dat")``
    * Load the results of a KBMOD run as trajectory objects:
        ``results = FileUtils.load_results_file_as_trajectories(
                      "kbmod/data/fake_results/results_DEMO.txt")``
    """

    @staticmethod
    def save_csv_from_list(file_name, data, overwrite=False):
        """Save a CSV file from a list of lists.

        Parameters
        ----------
        file_name : str
            The full path and file name for the result.
        data : list
            The data to save.
        overwrite : bool
            A Boolean indicating whether to overwrite the files
            or raise an exception on file existance.
        """
        if Path(file_name).is_file() and not overwrite:
            raise ValueError(f"{file_name} already exists")
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerows([x for x in data])

    @staticmethod
    def load_csv_to_list(file_name, use_dtype=None):
        """Load a CSV file to a list of numpy arrays.

        Parameters
        ----------
        file_name : str
            The full path and file name of the data.
        use_dtype : type
            The numpy array dtype to use.

        Returns
        -------
        data : list of numpy arrays
            The data loaded.
        """
        if not Path(file_name).is_file():
            raise ValueError(f"{file_name} does not exist")

        data = []
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(np.array(row, dtype=use_dtype))
        return data

    @staticmethod
    def load_time_dictionary(time_file):
        """Load a OrderedDict mapping ``visit_id`` to time stamp.

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
    def save_time_dictionary(time_file_name, time_mapping):
        """Save the mapping of visit_id -> time stamp to a file.

        Parameters
        ----------
        time_file_name : str
            The path and name of the time file.
        time_mapping : dict or OrderedDict
            The mapping of visit ID to time stamp.
        """
        with open(time_file_name, "w") as file:
            file.write("# visit_id mean_julian_date\n")
            for k in time_mapping.keys():
                file.write(f"{k} {time_mapping[k]}\n")

    @staticmethod
    def load_psf_dictionary(psf_file):
        """Load a OrderedDict mapping ``visit_id`` to PSF.

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
