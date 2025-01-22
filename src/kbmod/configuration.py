import math

from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from yaml import dump, safe_load
from kbmod.search import Logging


logger = Logging.getLogger(__name__)


class SearchConfiguration:
    """This class stores a collection of configuration parameter settings."""

    def __init__(self):
        self._required_params = set()

        self._params = {
            "center_thresh": 0.00,
            "chunk_size": 500000,
            "clip_negative": False,
            "cluster_eps": 20.0,
            "cluster_type": "all",
            "cluster_v_scale": 1.0,
            "coadds": [],
            "debug": False,
            "do_clustering": True,
            "do_mask": True,
            "do_stamp_filter": True,
            "encode_num_bytes": -1,
            "generator_config": {
                "name": "EclipticCenteredSearch",
                "velocities": [92.0, 526.0, 257],
                "angles": [-math.pi / 15, math.pi / 15, 129],
                "angle_units": "radian",
                "velocity_units": "pix / d",
                "given_ecliptic": None,
            },
            "gpu_filter": False,
            "im_filepath": None,
            "lh_level": 10.0,
            "max_lh": 1000.0,
            "mom_lims": [35.5, 35.5, 2.0, 0.3, 0.3],
            "num_obs": 10,
            "peak_offset": [2.0, 2.0],
            "psf_val": 1.4,
            "result_filename": None,
            "results_per_pixel": 8,
            "save_all_stamps": False,
            "sigmaG_lims": [25, 75],
            "stamp_radius": 10,
            "stamp_type": "sum",
            "track_filtered": False,
            "x_pixel_bounds": None,
            "x_pixel_buffer": None,
            "y_pixel_bounds": None,
            "y_pixel_buffer": None,
        }

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

        Raises
        ------
        Raises a ``ValueError`` if a parameter is missing.
        """
        for p in self._required_params:
            if self._params.get(p, None) is None:
                raise ValueError(f"Required configuration parameter {p} missing.")

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
