import warnings

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("Unable to determine the package version. " "This is likely a broken installation.")

# lazy import analysis to arrest
# loading large libraries in them
# import the filters subdirectory
from . import (
    analysis,
    analysis_utils,
    file_utils,
    filters,
    image_info,
    jointfit_functions,
    result_list,
    run_search,
)
