import math
import yaml
from pathlib import Path


class KBMODConfig:
    """This class stores a collection of configuration parameter settings."""

    def __init__(self):
        self._required_params = set(["im_filepath"])

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
            "im_filepath": None,
            "res_filepath": None,
            "time_file": None,
            "psf_file": None,
            "v_arr": [92.0, 526.0, 256],
            "ang_arr": [math.pi / 15, math.pi / 15, 128],
            "output_suffix": "search",
            "mjd_lims": None,
            "average_angle": None,
            "do_mask": True,
            "mask_num_images": 2,
            "mask_threshold": None,
            "mask_grow": 10,
            "lh_level": 10.0,
            "psf_val": 1.4,
            "num_obs": 10,
            "num_cores": 1,
            "visit_in_filename": [0, 6],
            "sigmaG_lims": [25, 75],
            "chunk_size": 500000,
            "max_lh": 1000.0,
            "center_thresh": 0.00,
            "peak_offset": [2.0, 2.0],
            "mom_lims": [35.5, 35.5, 2.0, 0.3, 0.3],
            "stamp_type": "sum",
            "stamp_radius": 10,
            "eps": 0.03,
            "gpu_filter": False,
            "do_clustering": True,
            "do_stamp_filter": True,
            "clip_negative": False,
            "cluster_type": "all",
            "cluster_function": "DBSCAN",
            "mask_bits_dict": default_mask_bits_dict,
            "flag_keys": default_flag_keys,
            "repeated_flag_keys": default_repeated_flag_keys,
            "bary_dist": None,
            "encode_psi_bytes": -1,
            "encode_phi_bytes": -1,
            "known_obj_thresh": None,
            "known_obj_jpl": False,
            "known_obj_obs": 3,
            "x_pixel_bounds": None,
            "y_pixel_bounds": None,
            "x_pixel_buffer": None,
            "y_pixel_buffer": None,
            "bary_dist": None,
            "debug": False,
        }

    def __getitem__(self, key):
        """Gets the value of a specific parameter.

        Parameters
        ----------
        key : str
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
        param : str
            The parameter name.
        value : any
            The parameter's value.
        strict : bool
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
        d : dict
            A dictionary mapping parameter name to valie.
        strict : bool
            Raise an exception on unknown parameters.

        Raises
        ------
        Raises a ``KeyError`` if the parameter is not part on the list of known parameters
        and ``strict`` is False.
        """
        for key, value in d.items():
            self.set(key, value, strict)

    def validate(self):
        """Check that the configuration has the necessary parameters.

        Raises
        ------
        Raises a ``ValueError`` if a parameter is missing.
        """
        for p in self._required_params:
            if self._params.get(p, None) is None:
                raise ValueError(f"Required configuration parameter {p} missing.")

    def load_from_file(self, filename, strict=True):
        """Load a configuration file and return the parameter dictionary.

        Parameters
        ----------
        filename : str
            The filename, including path, of the configuration file.
        strict : bool
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
            file_params = yaml.safe_load(config)

        # Merge in the new values.
        self.set_from_dict(file_params, strict)

        if strict:
            self.validate()

    def save_configuration(self, filename, overwrite=False):
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
            file.write(yaml.dump(self._params))
