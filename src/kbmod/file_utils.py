"""A collection of utility functions for working with files in KBMOD."""

import csv
import re
from collections import OrderedDict
from itertools import product
from math import copysign
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import *
from astropy.io import fits
from astropy.time import Time

import kbmod.search as kb
from kbmod.search import LayeredImage
from kbmod.trajectory_utils import trajectory_from_np_object


def load_deccam_layered_image(filename, psf):
    """Load a layered image from the legacy deccam format.

    Parameters
    ----------
    filename : `str`
        The name of the file to load.
    psf : `PSF`
        The PSF to use for the image.

    Returns
    -------
    img : `LayeredImage`
        The loaded image.

    Raises
    ------
    Raises a ``FileNotFoundError`` if the file does not exist.
    Raises a ``ValueError`` if any of the validation checks fail.
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"{filename} not found")

    img = None
    with fits.open(filename) as hdul:
        if len(hdul) < 4:
            raise ValueError("Not enough extensions for legacy deccam format")

        # Extract the obstime trying from a few keys and a few extensions.
        obstime = -1.0
        for key, ext in product(["MJD", "DATE-AVG", "MJD-OBS"], [0, 1]):
            if key in hdul[ext].header:
                value = hdul[ext].header[key]
                if type(value) is float:
                    obstime = value
                    break
                if type(value) is str:
                    timesys = hdul[ext].header.get("TIMESYS", "UTC").lower()
                    obstime = Time(value, scale=timesys).mjd
                    break

        img = LayeredImage(
            hdul[1].data.astype(np.float32),  # Science
            hdul[3].data.astype(np.float32),  # Variance
            hdul[2].data.astype(np.float32),  # Mask
            psf,
            obstime,
        )

    return img


class FileUtils:
    """A class of static methods for working with KBMOD files.

    Examples
    --------
    * Load an external file of visit ID to timestamp mappings.

    ``time_dict = FileUtils.load_time_dictionary("kbmod/data/demo_times.dat")``

    * Load the results of a KBMOD run as trajectory objects.

    ``FileUtils.load_results_file_as_trajectories("results_DEMO.txt")``
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
