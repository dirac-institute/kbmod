import math

from astropy.io import fits
from astropy.table import Table
from numpy import result_type
from pathlib import Path
import yaml
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
            "encode_num_bytes": -1,
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

    def set_multiple(self, overrides, strict=True):
        """Sets multiple parameters from a dictionary.

        Parameters
        ----------
        overrides : `dict`
            A dictionary of parameter->value to overwrite.
        strict : `bool`
            Raise an exception on unknown parameters.

        Raises
        ------
        Raises a ``KeyError`` if any parameter is not part on the list of known parameters
        and ``strict`` is False.
        """
        for key, value in overrides.items():
            self.set(key, value)

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

        # guaranteed to only have 1 element due to check above
        params = {col.name: safe_load(col.value[0]) for col in t.values()}
        return SearchConfiguration.from_dict(params)

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
        t = Table(hdu.data)
        return SearchConfiguration.from_table(t)

    @classmethod
    def from_file(cls, filename, strict=True):
        with open(filename) as ff:
            return SearchConfiguration.from_yaml(ff.read(), strict)

    def to_hdu(self):
        """Create a fits HDU with all the configuration parameters.

        Returns
        -------
        hdu : `astropy.io.fits.BinTableHDU`
            The HDU with the configuration information.
        """
        serialized_dict = {key: dump(val, default_flow_style=True) for key, val in self._params.items()}
        t = Table(
            rows=[
                serialized_dict,
            ]
        )
        return fits.table_to_hdu(t)

    def to_yaml(self):
        """Save a configuration file with the parameters.

        Returns
        -------
        result : `str`
            The serialized YAML string.
        """
        return dump(self._params)

    def to_file(self, filename, overwrite=False):
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
            file.write(self.to_yaml())
