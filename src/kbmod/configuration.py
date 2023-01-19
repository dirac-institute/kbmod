import copy
import math
import yaml
from pathlib import Path


class ConfigLoader:
    def __init__(self):
        self._required_params = set(["im_filepath", "res_filepath"])
        self._deprecated_params = set([])

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

        self._default_params = {
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
        }

    def check_required(self, params):
        """Check that a parameter dictionary has the required parameters.

        Parameters
        ----------
        params : dict
            A dictionary mapping parameter name (string) to value.

        Raises
        ------
        Raises a ValueError if any of the required parameters are missing.
        """
        for p in self._required_params:
            if p not in params:
                raise ValueError(f"Required configuration parameter {p} missing.")

    def filter_unused(self, params, verbose=True):
        """Create a copy of the parameter dictionary with deprecated or unspecified
        parameters removed.

        Parameters
        ----------
        params : dict
            A dictionary mapping parameter name (string) to value.
        verbose : bool
            Whether to produce warnings on filtered values.

        Returns
        -------
        filtered_params : dict
            A dictionary mapping parameter name (string) to value.
        """
        filtered_params = {}
        for p in params:
            if p in self._deprecated_params:
                if verbose:
                    print(f"Warning: Parameter '{p}' is deprecated and will not be used.")
                continue
            if p not in self._required_params and p not in self._default_params:
                if verbose:
                    print(f"Warning: Parameter '{p}' is unrecognized.")
                continue
            filtered_params[p] = params[p]
        return filtered_params

    def merge_defaults(self, params, verbose=True):
        """Create a copy of the parameter dictionary with any missing parameters
        set to their default value.

        Parameters
        ----------
        params : dict
            A dictionary mapping parameter name (string) to value.
        verbose : bool
            Whether to produce warnings on missing values.

        Returns
        -------
        final_params : dict
            A dictionary mapping parameter name (string) to value.
        """
        # Make a copy so we can use this in the save operation without modifying
        # the original.
        final_params = copy.copy(params)
        for p in self._default_params:
            if p not in final_params:
                if verbose:
                    print(f"Warning: Parameter '{p}' is missing. Adding default = {self._default_params[p]}")
                final_params[p] = self._default_params[p]
        return final_params

    def load_configuration(self, filename, verbose=True):
        """Load a configuration file and return the parameter dictionary.

        Parameters
        ----------
        filename : str
            The filename, including path, of the configuration file.
        verbose : bool
            Indicates whether to display verbose output.

        Returns
        -------
        final_params : dict
            A dictionary mapping parameter name (string) to value.

        Raises
        ------
        Raises a ValueError if the configuration file is not found.
        """
        if not Path(filename).is_file():
            raise ValueError(f"Configuration file {filename} not found.")

        # Read the user-specified parameters from the file.
        file_params = {}
        if verbose:
            print(f"Loading configuration from: {filename}")
        with open(filename, "r") as config:
            file_params = yaml.safe_load(config)

        # Check required parameters and filter deprecated parameters.
        self.check_required(file_params)
        filtered_params = self.filter_unused(file_params, verbose)

        # Load any default params that were not included in the user's file.
        final_params = self.merge_defaults(filtered_params, verbose)

        return final_params

    def save_configuration(self, params, filename, overwrite=False):
        """Save a configuration file and return the parameter dictionary.

        Parameters
        ----------
        params : dict
            A dictionary mapping parameter name (string) to value.
        filename : str
            The filename, including path, of the configuration file.
        overwrite : bool
            Indicates whether to overwrite an existing file.
        """
        if Path(filename).is_file() and not overwrite:
            print(f"Warning: Configuration file {filename} already exists.")
            return

        # Check required parameters and filter out unused parameters.
        self.check_required(params)
        filtered_params = self.filter_unused(params, False)

        # Load any default params that were not included in the dictionary.
        final_params = self.merge_defaults(params, False)

        with open(filename, "w") as file:
            file.write(yaml.dump(final_params))
