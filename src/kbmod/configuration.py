import copy
import math

from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from yaml import dump, safe_load
from kbmod.search import Logging


logger = Logging.getLogger(__name__)


class _ParamInfo:
    """Class to store information about a configuration parameter.

    Parameters
    ----------
    name : `str`
        The parameter name.
    default_value : any, optional
        The default value for the parameter. If not provided, defaults to None.
    description : `str`, optional
        A description of the parameter. If not provided, defaults to an empty string.
    section : `str`, optional
        The section the parameter belongs to. If not provided, defaults to "other".
    validate_func : `callable`, optional
        A function to validate the parameter's value. If not provided, defaults to None.
    required : `bool`, optional
        Whether the parameter is required. If not provided, defaults to False.
    """

    def __init__(
        self,
        name,
        default_value,
        description="",
        section="other",
        validate_func=None,
        required=False,
    ):
        self.name = name
        self.default_value = default_value
        self.description = description
        self.section = section
        self.validate_func = validate_func
        self.required = required

    def __str__(self):
        return f"Parameter {self.name}: {self.description} (Default: {self.default_value})"

    def validate(self, value):
        """Validate the parameter's value using the provided validation function.

        Parameters
        ----------
        value : any
            The value to validate.

        Returns
        -------
        `bool`
            True if the value is valid, False otherwise.
        """
        if self.required and value is None:
            return False
        if self.validate_func is not None:
            return self.validate_func(value)
        return True


# List of all the supported configuration parameters (in alphabetical order).
_SUPPORTED_PARAMS = [
    _ParamInfo(
        name="clip_negative",
        default_value=False,
        description="If True remove all negative values prior to sigmaG computing the percentiles.",
        section="filtering",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="cluster_eps",
        default_value=20.0,
        description="The epsilon parameter for clustering (in pixels).",
        section="clustering",
        validate_func=lambda x: isinstance(x, (int, float)) and x >= 0,
    ),
    _ParamInfo(
        name="cluster_type",
        default_value="all",
        description="The type of clustering algorithm to use (if do_clustering = True).",
        section="clustering",
        validate_func=lambda x: isinstance(x, str),
    ),
    _ParamInfo(
        name="cluster_v_scale",
        default_value=1.0,
        description="The weight of differences in velocity relative to differences in distances during clustering.",
        section="clustering",
        validate_func=lambda x: isinstance(x, (int, float)) and x >= 0,
    ),
    _ParamInfo(
        name="cnn_filter",
        default_value=False,
        description="If True, applies a CNN filter to the stamps.",
        section="filtering",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="cnn_model",
        default_value=None,
        description="The path to the CNN model file to use for filtering.",
        section="filtering",
        validate_func=lambda x: isinstance(x, str) or x is None,
    ),
    _ParamInfo(
        name="cnn_coadd_type",
        default_value="mean",
        description="The type of coadd to use for CNN filtering ('mean', 'median', or 'sum').",
        section="filtering",
        validate_func=lambda x: x in ["mean", "median", "sum"],
    ),
    _ParamInfo(
        name="cnn_stamp_radius",
        default_value=49,
        description="The radius (in pixels) of the stamp to use for CNN filtering.",
        section="filtering",
        validate_func=lambda x: isinstance(x, int) and x > 0,
    ),
    _ParamInfo(
        name="cnn_model_type",
        default_value="resnet18",
        description="The type of CNN model to use ('resnet18', 'resnet34', etc.).",
        section="filtering",
        validate_func=lambda x: isinstance(x, str),
    ),
    _ParamInfo(
        name="coadds",
        default_value=[],
        description="The list of coadd images to compute ('mean', 'median', 'sum', 'weighted').",
        section="stamps",
        validate_func=lambda x: isinstance(x, list) and all(isinstance(i, int) for i in x),
    ),
    _ParamInfo(
        name="compute_ra_dec",
        default_value=True,
        description="If True, compute RA and Dec for each result.",
        section="output",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="cpu_only",
        default_value=False,
        description="If True, only use the CPU for processing, even if a GPU is available.",
        section="other",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="debug",
        default_value=False,
        description="Run with debug logging enabled.",
        section="other",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="do_clustering",
        default_value=True,
        description="If true, perform clustering on the results.",
        section="clustering",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="drop_columns",
        default_value=[],
        description="List of result table columns to drop.",
        section="output",
        validate_func=lambda x: isinstance(x, list) and all(isinstance(i, str) for i in x),
    ),
    _ParamInfo(
        name="encode_num_bytes",
        default_value=-1,
        description="Number of bytes to use for encoding pixel values on GPU. -1 means no encoding.",
        section="core",
        validate_func=lambda x: x in set([-1, 1, 2, 4]),
    ),
    _ParamInfo(
        name="generator_config",
        default_value={
            "name": "EclipticCenteredSearch",
            "velocities": [92.0, 526.0, 257],
            "angles": [-math.pi / 15, math.pi / 15, 129],
            "angle_units": "radian",
            "velocity_units": "pix / d",
            "given_ecliptic": None,
        },
        description="Configuration dictionary for the trajectory generator.",
        section="core",
        validate_func=lambda x: isinstance(x, dict) and "name" in x,
    ),
    _ParamInfo(
        name="generate_psi_phi",
        default_value=True,
        description="If True, computes the psi and phi curves and saves them with the results.",
        section="filtering",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="gpu_filter",
        default_value=True,
        description="If True, performs initial sigmaG filtering on GPU.",
        section="filtering",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="lh_level",
        default_value=10.0,
        description="The log-likelihood level above which results are kept.",
        section="filtering",
        validate_func=lambda x: isinstance(x, (int, float)),
    ),
    _ParamInfo(
        name="max_masked_pixels",
        default_value=0.5,
        description="The maximum fraction of masked pixels allowed before an input image is dropped.",
        section="core",
        validate_func=lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
    ),
    _ParamInfo(
        name="max_results",
        default_value=100_000,
        description="The maximum number of results to save after all filtering.",
        section="filtering",
        validate_func=lambda x: isinstance(x, int),
    ),
    _ParamInfo(
        name="near_dup_thresh",
        default_value=10,
        description="The threshold for considering two observations as near duplicates (in pixels).",
        section="filtering",
        validate_func=lambda x: isinstance(x, int),
    ),
    _ParamInfo(
        name="nightly_coadds",
        default_value=False,
        description="If True, generate an additional coadd for each calendar date.",
        section="stamps",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="num_obs",
        default_value=10,
        description="The minimum number of valid observations for the trajectory to be accepted.",
        section="filtering",
        validate_func=lambda x: isinstance(x, int),
    ),
    _ParamInfo(
        name="peak_offset_max",
        default_value=None,
        description="Maximum allowed offset (in pixels) between predicted and detected peak positions.",
        section="filtering",
        validate_func=lambda x: isinstance(x, (int, float)) or x is None,
    ),
    _ParamInfo(
        name="pred_line_cluster",
        default_value=False,
        description="If True, applies line clustering to the predicted lines.",
        section="filtering",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="pred_line_params",
        default_value=[4.0, 2, 60],
        description="Parameters for the line prediction model.",
        section="filtering",
        validate_func=lambda x: isinstance(x, list) and len(x) == 3,
    ),
    _ParamInfo(
        name="psf_val",
        default_value=1.4,
        description="The default standard deviation of the Gaussian PSF in pixels (if not provided in the data).",
        section="core",
        validate_func=lambda x: isinstance(x, (int, float)) and x > 0.0,
    ),
    _ParamInfo(
        name="result_filename",
        default_value=None,
        description="The filename to which results will be saved.",
        section="core",
        validate_func=lambda x: isinstance(x, str) or x is None,
    ),
    _ParamInfo(
        name="results_per_pixel",
        default_value=8,
        description="The maximum number of results to return from the GPU per pixel.",
        section="filtering",
        validate_func=lambda x: isinstance(x, int) and x > 0,
    ),
    _ParamInfo(
        name="save_all_stamps",
        default_value=False,
        description="If True, save all stamps to the results.",
        section="output",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="save_config",
        default_value=True,
        description="If True, save the configuration used for processing.",
        section="output",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="separate_col_files",
        default_value=["all_stamps"],
        description="List of columns to save in separate files.",
        section="output",
        validate_func=lambda x: isinstance(x, list) and all(isinstance(i, str) for i in x),
    ),
    _ParamInfo(
        name="sigmaG_filter",
        default_value=True,
        description="If True, apply sigmaG filtering.",
        section="filtering",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="sigmaG_lims",
        default_value=[25, 75],
        description="The lower and upper limits for sigmaG filtering.",
        section="filtering",
        validate_func=lambda x: len(x) == 2 and x[0] < x[1] and all(isinstance(i, (int, float)) for i in x),
    ),
    _ParamInfo(
        name="stamp_radius",
        default_value=10,
        description="The radius (in pixels) of the stamp to extract.",
        section="stamps",
        validate_func=lambda x: isinstance(x, int) and x > 0,
    ),
    _ParamInfo(
        name="stamp_type",
        default_value="sum",
        description="The type of stamp to extract.",
        section="stamps",
        validate_func=lambda x: x in ["sum", "mean", "median", "weighted"],
    ),
    _ParamInfo(
        name="track_filtered",
        default_value=False,
        description="If True, track the filtered objects in the results table.",
        section="filtering",
        validate_func=lambda x: isinstance(x, bool),
    ),
    _ParamInfo(
        name="x_pixel_bounds",
        default_value=None,
        description="The x pixel bounds for the search starting location (None = use every pixel).",
        section="core",
        validate_func=lambda x: x is None or (len(x) == 2 and x[0] < x[1]),
    ),
    _ParamInfo(
        name="x_pixel_buffer",
        default_value=None,
        description="If not None, the number of x pixels beyond the image bounds to use for starting coordinates.",
        section="core",
        validate_func=lambda x: x is None or (isinstance(x, int) and x >= 0),
    ),
    _ParamInfo(
        name="y_pixel_bounds",
        default_value=None,
        description="The y pixel bounds for the search starting location (None = use every pixel).",
        section="core",
        validate_func=lambda x: x is None or (len(x) == 2 and x[0] < x[1]),
    ),
    _ParamInfo(
        name="y_pixel_buffer",
        default_value=None,
        description="If not None, the number of y pixels beyond the image bounds to use for starting coordinates.",
        section="core",
        validate_func=lambda x: x is None or (isinstance(x, int) and x >= 0),
    ),
]


class SearchConfiguration:
    """This class stores a collection of configuration parameter settings.

    Parameters
    ----------
    data : `dict`
        A dictionary of initial values.
    """

    def __init__(self, data=None):
        # Reprocess the list of supported parameters into dictionaries for easy access.
        self._param_info = {p.name: p for p in _SUPPORTED_PARAMS}
        self._params = {p.name: p.default_value for p in _SUPPORTED_PARAMS}
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
        # Check parameters that have known constraints.
        for key, value in self._params.items():
            param_info = self._param_info.get(key, None)
            if param_info is not None and not param_info.validate(value):
                logger.warning(f"Invalid value for parameter {key}: {value}")
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
        logger.info(f"Saving configuration to {filename}")

        # Output the configuration file in sections for easier reading. We add the sections
        # in the order we want them to appear.
        section_to_params = {
            "core": [],
            "filtering": [],
            "stamps": [],
            "clustering": [],
            "output": [],
            "other": [],
        }
        for key, value in self._param_info.items():
            section = value.section if value.section in section_to_params else "other"
            section_to_params[section].append(key)

        with open(filename, "w") as file:
            for section, section_keys in section_to_params.items():
                file.write("# ======================================================================\n")
                file.write(f"# {section.capitalize()} Configuration\n")
                file.write("# ======================================================================\n")
                for key in section_keys:
                    if key in self._param_info and self._param_info[key].description:
                        file.write(f"\n# {self._param_info[key].description}\n")
                        file.write(dump({key: self._params[key]}))
                file.write("\n")
