"""A collection of utility functions for working with files in KBMOD."""

import csv
import re
from collections import OrderedDict
from math import copysign
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import *
from astropy.time import Time

import kbmod.search as kb
from kbmod.trajectory_utils import trajectory_from_np_object


class FileUtils:
    """A class of static methods for working with KBMOD files.

    Examples
    --------
    * Load an external file of visit ID to timestamp mappings.

    ``time_dict = FileUtils.load_time_dictionary("kbmod/data/demo_times.dat")``

    * Load the results of a KBMOD run as trajectory objects.

    ``FileUtils.load_results_file_as_trajectories("results_DEMO.txt")``

    * Make a filename safe.

    ``FileUtils.make_safe_filename("my string, is here")``
    """

    @staticmethod
    def make_safe_filename(s):
        """Makes a safe file name out of an arbitrary string.

        Preserves the separators (spaces, commas, tabs, etc.) with underscores
        and removes all other non-alphanumeric characters.

        Parameters
        ----------
        s : string
            The input string

        Returns
        -------
        res : string
            The output string
        """
        separators = set([" ", ".", ",", ";", "\t", "\n", ":", "-", "|", "/"])

        # If the character is a letter or number, keep it.
        # If it is a separator, replace with "_".
        # Otherwise discard it.
        pick_char = lambda x: x if (x.isalnum() or x == "_") else ("_" if x in separators else "")
        res = "".join(pick_char(x) for x in s)
        return res

    @staticmethod
    def visit_from_file_name(filename):
        """Automatically extract the visit ID from the file name.

        Uses the heuristic that the visit ID is the first numeric
        string of at least length 5 digits in the file name.

        Parameters
        ----------
        filename : str
            The file name

        Returns
        -------
        result : str
            The visit ID string or None if there is no match.
        """
        expr = re.compile(r"\d{4}(?:\d+)")
        res = expr.search(filename)
        if res is None:
            return None
        return res.group()

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
    def load_csv_to_list(file_name, use_dtype=None, none_if_missing=False):
        """Load a CSV file to a list of numpy arrays.

        Parameters
        ----------
        file_name : str
            The full path and file name of the data.
        use_dtype : type
            The numpy array dtype to use.
        none_if_missing : bool
            Return None if the file is missing. The default is to
            raise an exception if the file is missing.

        Returns
        -------
        data : list of numpy arrays
            The data loaded.
        """
        if not Path(file_name).is_file():
            if none_if_missing:
                return None
            else:
                raise FileNotFoundError

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
    def save_results_file(filename, results):
        """Save the result trajectories to a file.

        Parameters
        ----------
        filename : str
            The filename of the results.
        results : list
             list of trajectory objects.
        """
        np.savetxt(filename, results, fmt="%s")

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
            ndmin=2,
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
        results = [trajectory_from_np_object(x) for x in np_results]
        return results

    @staticmethod
    def mpc_reader(filename):
        """Read in a file with observations in MPC format and return the coordinates.

        Parameters
        ----------
        filename: str
            The name of the file with the MPC-formatted observations.

        Returns
        -------
        coords: astropy SkyCoord object
            A SkyCoord object with the ra, dec of the observations.
        times: astropy Time object
            Times of the observations
        """
        iso_times = []
        time_frac = []
        ra = []
        dec = []

        with open(filename, "r") as f:
            for line in f:
                year = str(line[15:19])
                month = str(line[20:22])
                day = str(line[23:25])
                iso_times.append(str("%s-%s-%s" % (year, month, day)))
                time_frac.append(str(line[25:31]))
                ra.append(str(line[32:44]))
                dec.append(str(line[44:56]))

        coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        t = Time(iso_times)
        t_obs = []
        for t_i, frac in zip(t, time_frac):
            t_obs.append(t_i.mjd + float(frac))
        obs_times = Time(t_obs, format="mjd")

        return coords, obs_times

    @staticmethod
    def format_result_mpc(coords, t, observatory="X05"):
        """
        This method will take a single result in and return a corresponding
        MPC formatted string.

        Parameters
        ----------
        coords : SkyCoord
            The sky coordinates of the observation.
        t : Time
            The time of the observation as an astropy Time object.
        observatory : string
            The three digit observatory code to use.

        Returns
        -------
        mpc_line: string
            An MPC-formatted string of the observation
        """
        mjd_frac = t.mjd % 1.0
        ra_hms = coords.ra.hms
        dec_dms = coords.dec.dms

        if dec_dms.d == 0:
            if copysign(1, dec_dms.d) == -1.0:
                dec_dms_d = "-00"
            else:
                dec_dms_d = "+00"
        else:
            dec_dms_d = "%+03i" % dec_dms.d

        mpc_line = "     c111112  c%4i %02i %08.5f %02i %02i %06.3f%s %02i %05.2f                     %s" % (
            t.datetime.year,
            t.datetime.month,
            t.datetime.day + mjd_frac,
            ra_hms.h,
            ra_hms.m,
            ra_hms.s,
            dec_dms_d,
            np.abs(dec_dms.m),
            np.abs(dec_dms.s),
            observatory,
        )
        return mpc_line

    @staticmethod
    def save_results_mpc(file_out, coords, times, observatory="X05"):
        """
        Save the MPC-formatted observations to file.

        Parameters
        ----------
        file_out: str
            The output filename with the MPC-formatted observations
            of the KBMOD search result.
        coords : list of SkyCoord
            A list of sky coordinates (SkyCoord objects) of the observation.
        t : list of Time
            A list of times for each observation.
        observatory : string
            The three digit observatory code to use.
        """
        if len(times) != len(coords):
            raise ValueError(f"Unequal lists {len(times)} != {len(coords)}")

        with open(file_out, "w") as f:
            for i in range(len(times)):
                mpc_line = FileUtils.format_result_mpc(coords[i], times[i], observatory)
                f.write(mpc_line + "\n")
