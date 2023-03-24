import warnings

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("Unable to determine the package version. " "This is likely a broken installation.")

from . import run_search
from . import image_info
from . import analysis_utils
from . import jointfit_functions
from . import result_list
from . import file_utils

# import the filters subdirectory
from . import filters

# lazy import analysis to arrest
# loading large libraries in them
from . import analysis
