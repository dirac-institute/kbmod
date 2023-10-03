import math

from astropy.io import fits
from astropy.table import Table
from numpy import result_type
from pathlib import Path
import pickle
from yaml import dump, safe_load


class SearchConfiguration:
    """This class stores a collection of configuration parameter settings."""

    def __init__(self):
        self._required_params = set()

        default_mask_bits_dict = {
            "BAD": 0,
            "CLIPPED": 9,
            "CR": 3,
            "CROSSTALK": 10,
            "DETECTED": 5,
            "DETECTED_NEGATIVE": 6,
            "EDGE": 4,
            "INEXACT_PSF": 11,
            "INTRP": 2,
            "NOT_DEBLENDED": 12,
            "NO_DATA": 8,
            "REJECTED": 13,
            "SAT": 1,
            "SENSOR_EDGE": 14,
            "SUSPECT": 7,
            "UNMASKEDNAN": 15,
        }
        default_flag_keys = ["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]
        default_repeated_flag_keys = []

        self._params = {
            "ang_arr": [math.pi / 15, math.pi / 15, 128],
            "average_angle": None,
            "center_thresh": 0.00,
            "chunk_size": 500000,
            "clip_negative": False,
            "cluster_function": "DBSCAN",
            "cluster_type": "all",
            "debug": False,
            "do_clustering": True,
            "do_mask": True,
            "do_stamp_filter": True,
            "eps": 0.03,
            "encode_psi_bytes": -1,
            "encode_phi_bytes": -1,
            "flag_keys": default_flag_keys,
            "gpu_filter": False,
            "im_filepath": None,
            "known_obj_obs": 3,
            "known_obj_thresh": None,
            "known_obj_jpl": False,
            "lh_level": 10.0,
            "mask_bits_dict": default_mask_bits_dict,
            "mask_bit_vector": None,
            "mask_grow": 10,
            "mask_num_images": 2,
            "mask_threshold": None,
            "max_lh": 1000.0,
            "mjd_lims": None,
            "mom_lims": [35.5, 35.5, 2.0, 0.3, 0.3],
            "num_cores": 1,
            "num_obs": 10,
            "output_suffix": "search",
            "peak_offset": [2.0, 2.0],
            "psf_val": 1.4,
            "psf_file": None,
            "repeated_flag_keys": default_repeated_flag_keys,
            "res_filepath": None,
            "sigmaG_lims": [25, 75],
            "stamp_radius": 10,
            "stamp_type": "sum",
            "time_file": None,
            "v_arr": [92.0, 526.0, 256],
            "x_pixel_bounds": None,
            "x_pixel_buffer": None,
            "y_pixel_bounds": None,
            "y_pixel_buffer": None,
        }

    def __getitem__(self, key):
        """Gets the value of a specific parameter.

        Parameters
        ----------
        key : `str`
            The parameter name.

        Raises
        ------
        Raises a KeyError if the parameter is not included.
        """
        return self._params[key]

    def set(self, param, value, strict=True):
        """Sets the value of a specific parameter.

        Parameters
        ----------
        param : `str`
            The parameter name.
        value : any
            The parameter's value.
        strict : `bool`
            Raise an exception on unknown parameters.

        Raises
        ------
        Raises a ``KeyError`` if the parameter is not part on the list of known parameters
        and ``strict`` is False.
        """
        if param not in self._params:
            if strict:
                raise KeyError(f"Invalid parameter: {param}")
            else:
                print(f"Ignoring invalid parameter: {param}")
        else:
            self._params[param] = value

    def set_from_dict(self, d, strict=True):
        """Sets multiple values from a dictionary.

        Parameters
        ----------
        d : `dict`
            A dictionary mapping parameter name to valie.
        strict : `bool`
            Raise an exception on unknown parameters.

        Raises
        ------
        Raises a ``KeyError`` if the parameter is not part on the list of known parameters
        and ``strict`` is False.
        """
        for key, value in d.items():
            self.set(key, value, strict)

    def set_from_table(self, t, strict=True):
        """Sets multiple values from an astropy Table with a single row and
        one column for each parameter.

        Parameters
        ----------
        t : `~astropy.table.Table`
            Astropy Table containing the required configuration parameters.
        strict : `bool`
            Raise an exception on unknown parameters.

        Raises
        ------
        Raises a ``KeyError`` if the parameter is not part on the list of known parameters
        and ``strict`` is False.

        Raises a ``ValueError`` if the table is the wrong shape.
        """
        if len(t) > 1:
            raise ValueError(f"More than one row in the configuration table ({len(t)}).")
        for key in t.colnames:
            # We use a special indicator for serializing certain types (including
            # None and dict) to FITS.
            if key.startswith("__PICKLED_"):
                val = pickle.loads(t[key].value[0])
                key = key[10:]
            else:
                val = t[key][0]

            self.set(key, val, strict)

    def to_table(self, make_fits_safe=False):
        """Create an astropy table with all the configuration parameters.

        Parameter
        ---------
        make_fits_safe : `bool`
            Override Nones and dictionaries so we can write to FITS.

        Returns
        -------
        t: `~astropy.table.Table`
            The configuration table.
        """
        t = Table()
        for col in self._params.keys():
            val = self._params[col]
            t[col] = [val]

            # If Table does not understand the type, pickle it.
            if make_fits_safe and t[col].dtype == "O":
                t.remove_column(col)
                t["__PICKLED_" + col] = pickle.dumps(val)

        return t

    def validate(self):
        """Check that the configuration has the necessary parameters.

        Raises
        ------
        Raises a ``ValueError`` if a parameter is missing.
        """
        for p in self._required_params:
            if self._params.get(p, None) is None:
                raise ValueError(f"Required configuration parameter {p} missing.")

    def load_from_yaml_file(self, filename, strict=True):
        """Load a configuration from a YAML file.

        Parameters
        ----------
        filename : `str`
            The filename, including path, of the configuration file.
        strict : `bool`
            Raise an exception on unknown parameters.

        Raises
        ------
        Raises a ``ValueError`` if the configuration file is not found.
        Raises a ``KeyError`` if the parameter is not part on the list of known parameters
        and ``strict`` is False.
        """
        if not Path(filename).is_file():
            raise ValueError(f"Configuration file {filename} not found.")

        # Read the user-specified parameters from the file.
        file_params = {}
        with open(filename, "r") as config:
            file_params = safe_load(config)

        # Merge in the new values.
        self.set_from_dict(file_params, strict)

        if strict:
            self.validate()

    def load_from_fits_file(self, filename, layer=0, strict=True):
        """Load a configuration from a FITS extension file.

        Parameters
        ----------
        filename : `str`
            The filename, including path, of the configuration file.
        layer : `int`
            The extension number to use.
        strict : `bool`
            Raise an exception on unknown parameters.

        Raises
        ------
        Raises a ``ValueError`` if the configuration file is not found.
        Raises a ``KeyError`` if the parameter is not part on the list of known parameters
        and ``strict`` is False.
        """
        if not Path(filename).is_file():
            raise ValueError(f"Configuration file {filename} not found.")

        # Read the user-specified parameters from the file.
        t = Table.read(filename, hdu=layer)
        self.set_from_table(t)

        if strict:
            self.validate()

    def save_to_yaml_file(self, filename, overwrite=False):
        """Save a configuration file with the parameters.

        Parameters
        ----------
        filename : str
            The filename, including path, of the configuration file.
        overwrite : bool
            Indicates whether to overwrite an existing file.
        """
        if Path(filename).is_file() and not overwrite:
            print(f"Warning: Configuration file {filename} already exists.")
            return

        with open(filename, "w") as file:
            file.write(dump(self._params))

    def append_to_fits(self, filename):
        """Append the configuration table as a new extension on a FITS file
        (creating a new file if needed).

        Parameters
        ----------
        filename : str
            The filename, including path, of the configuration file.
        """
        t = self.to_table(make_fits_safe=True)
        t.write(filename, append=True)
