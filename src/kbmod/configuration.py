import ast
import math

from astropy.io import fits
from astropy.table import Table
from numpy import result_type
from pathlib import Path
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

    def validate(self):
        """Check that the configuration has the necessary parameters.

        Raises
        ------
        Raises a ``ValueError`` if a parameter is missing.
        """
        for p in self._required_params:
            if self._params.get(p, None) is None:
                raise ValueError(f"Required configuration parameter {p} missing.")

    @classmethod
    def from_dict(cls, d, strict=True):
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
        config = SearchConfiguration()
        for key, value in d.items():
            config.set(key, value, strict)
        return config

    @classmethod
    def from_table(cls, t, strict=True):
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

        config = SearchConfiguration()
        for key in t.colnames:
            # We use a special indicator for serializing certain types (including
            # None and dict) to FITS.
            if key.startswith("__NONE__"):
                val = None
                key = key[8:]
            elif key.startswith("__DICT__"):
                val = dict(t[key][0])
                key = key[8:]
            else:
                val = t[key][0]

            config.set(key, val, strict)
        return config

    @classmethod
    def from_yaml(cls, config, strict=True):
        """Load a configuration from a YAML file.

        Parameters
        ----------
        config : `str` or `_io.TextIOWrapper`
            The serialized YAML data.
        strict : `bool`
            Raise an exception on unknown parameters.

        Raises
        ------
        Raises a ``KeyError`` if the parameter is not part on the list of known parameters
        and ``strict`` is False.
        """
        yaml_params = safe_load(config)
        return SearchConfiguration.from_dict(yaml_params, strict)

    @classmethod
    def from_hdu(cls, hdu, strict=True):
        """Load a configuration from a FITS extension file.

        Parameters
        ----------
        hdu : `astropy.io.fits.BinTableHDU`
            The HDU from which to parse the configuration information.
        strict : `bool`
            Raise an exception on unknown parameters.

        Raises
        ------
        Raises a ``KeyError`` if the parameter is not part on the list of known parameters
        and ``strict`` is False.
        """
        config = SearchConfiguration()
        for column in hdu.data.columns:
            key = column.name
            val = hdu.data[key][0]

            # We use a special indicator for serializing certain types (including
            # None and dict) to FITS.
            if type(val) is str and val == "__NONE__":
                val = None
            elif key.startswith("__DICT__"):
                val = ast.literal_eval(val)
                key = key[8:]

            config.set(key, val, strict)
        return config

    @classmethod
    def from_file(cls, filename, extension=0, strict=True):
        if filename.endswith("yaml"):
            with open(filename) as ff:
                return SearchConfiguration.from_yaml(ff.read())
        elif ".fits" in filename:
            with fits.open(filename) as ff:
                return SearchConfiguration.from_hdu(ff[extension])
        raise ValueError("Configuration file suffix unrecognized.")

    def to_hdu(self):
        """Create a fits HDU with all the configuration parameters.

        Returns
        -------
        hdu : `astropy.io.fits.BinTableHDU`
            The HDU with the configuration information.
        """
        t = Table()
        for col in self._params.keys():
            val = self._params[col]
            if val is None:
                t[col] = ["__NONE__"]
            elif type(val) is dict:
                t["__DICT__" + col] = [str(val)]
            else:
                t[col] = [val]
        return fits.table_to_hdu(t)

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
