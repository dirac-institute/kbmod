import warnings

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("Unable to determine the package version. " "This is likely a broken installation.")

from . import kbmodpy
from . import run_search
from . import image_info
from . import kbmod_info
from . import evaluate
from . import analysis_utils
from . import jointfit_functions

# lazy import analysis to arrest
# loading large libraries in them
from . import analysis
