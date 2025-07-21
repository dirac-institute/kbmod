import copy
import math

from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from yaml import dump, safe_load
from kbmod.search import Logging


logger = Logging.getLogger(__name__)


class SearchConfiguration:
    """This class stores a collection of configuration parameter settings.

    Parameters
    ----------
    data : `dict`
        A dictionary of initial values.
    """

    def __init__(self, data=None):
        self._required_params = set()

        self._params = {
            "clip_negative": False,
            "cluster_eps": 20.0,
            "cluster_type": "all",
            "cluster_v_scale": 1.0,
            "coadds": [],
            "compute_ra_dec": True,
            "cpu_only": False,
            "debug": False,
            "do_clustering": True,
            "drop_columns": [],
            "encode_num_bytes": -1,
            "generator_config": {
                "name": "EclipticCenteredSearch",
                "velocities": [92.0, 526.0, 257],
                "angles": [-math.pi / 15, math.pi / 15, 129],
                "angle_units": "radian",
                "velocity_units": "pix / d",
                "given_ecliptic": None,
            },
            "generate_psi_phi": True,
            "gpu_filter": False,
            "lh_level": 10.0,
            "max_results": 100_000,
            "near_dup_thresh": 10,
            "nightly_coadds": False,
            "num_obs": 10,
            "psf_val": 1.4,
            "result_filename": None,
            "results_per_pixel": 8,
            "save_all_stamps": False,
            "save_config": True,
            "separate_col_files": ["all_stamps"],
            "sigmaG_filter": True,
            "sigmaG_lims": [25, 75],
            "stamp_radius": 10,
            "stamp_type": "sum",
            "track_filtered": False,
            "x_pixel_bounds": None,
            "x_pixel_buffer": None,
            "y_pixel_bounds": None,
            "y_pixel_buffer": None,
            "cnn_filter": False,
            "cnn_model": None,
            "cnn_coadd_type": "mean",
            "cnn_stamp_radius": 49,
            "cnn_model_type": "resnet18",
            "peak_offset_max": None,
        }

        if data is not None:
            self.set_multiple(data)

    def __contains__(self, key):
        return key in self._params

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

    def __str__(self):
        result = "Configuration:\n"
        for key, value in self._params.items():
            result += f"{key}: {value}\n"
        return result

    def copy(self):
        """Create a new deep copy of the configuration."""
        return copy.deepcopy(self)

    def set(self, param, value, warn_on_unknown=False):
        """Sets the value of a specific parameter.

        Parameters
        ----------
        param : `str`
            The parameter name.
        value : any
            The parameter's value.
        warn_on_unknown : `bool`
            Generate a warning if the parameter is not known.
        """
        if warn_on_unknown and param not in self._params:
            logger.warning(f"Setting unknown parameter: {param}")
        self._params[param] = value

    def set_multiple(self, overrides):
        """Sets multiple parameters from a dictionary.

        Parameters
        ----------
        overrides : `dict`
            A dictionary of parameter->value to overwrite.
        """
        for key, value in overrides.items():
            self.set(key, value)

    def validate(self):
        """Check that the configuration has the necessary parameters.

        Returns
        -------
        `bool`
            Returns True if the configuration is valid and False (logging the reason)
            if the configuration is invalid.
        """
        for p in self._required_params:
            if self._params.get(p, None) is None:
                logger.warning(f"Required configuration parameter {p} missing.")
                return False

        # Check parameters that have known constraints.
        if self._params["results_per_pixel"] <= 0:
            logger.warning(f"Invalid results_per_pixel: {self._params['results_per_pixel']}")
            return False

        if self._params["encode_num_bytes"] not in set([-1, 1, 2, 4]):
            logger.warning(
                f"Invalid encode_num_bytes: {self._params['encode_num_bytes']} "
                "must be one of -1, 1, 2, or 4."
            )
            return False

        if self._params["psf_val"] <= 0.0:
            logger.warning(f"Invalid psf_val {self._params['psf_val']}")
            return False

        if self._params["x_pixel_bounds"] is not None:
            if len(self._params["x_pixel_bounds"]) != 2:
                logger.warning(f"Expected two values for x_pixel_bounds")
                return False
            if self._params["x_pixel_bounds"][1] <= self._params["x_pixel_bounds"][0]:
                logger.warning(f"Invalid x_pixel_bounds: {self._params['x_pixel_bounds']}")
                return False

        if self._params["y_pixel_bounds"] is not None:
            if len(self._params["y_pixel_bounds"]) != 2:
                logger.warning(f"Expected two values for y_pixel_bounds")
                return False
            if self._params["y_pixel_bounds"][1] <= self._params["y_pixel_bounds"][0]:
                logger.warning(f"Invalid y_pixel_bounds: {self._params['y_pixel_bounds']}")
                return False

        return True

    @classmethod
    def from_dict(cls, d):
        """Sets multiple values from a dictionary.

        Parameters
        ----------
        d : `dict`
            A dictionary mapping parameter name to valie.
        """
        config = SearchConfiguration()
        for key, value in d.items():
            config.set(key, value)
        return config

    @classmethod
    def from_table(cls, t):
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
    def from_yaml(cls, config):
        """Load a configuration from a YAML file.

        Parameters
        ----------
        config : `str` or `_io.TextIOWrapper`
            The serialized YAML data.
        """
        yaml_params = safe_load(config)
        return SearchConfiguration.from_dict(yaml_params)

    @classmethod
    def from_hdu(cls, hdu):
        """Load a configuration from a FITS extension file.

        Parameters
        ----------
        hdu : `astropy.io.fits.BinTableHDU`
            The HDU from which to parse the configuration information.
        """
        t = Table(hdu.data)
        return SearchConfiguration.from_table(t)

    @classmethod
    def from_file(cls, filename):
        with open(filename) as ff:
            return SearchConfiguration.from_yaml(ff.read())

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
            logger.warning(f"Configuration file {filename} already exists.")
            return

        with open(filename, "w") as file:
            file.write(self.to_yaml())
