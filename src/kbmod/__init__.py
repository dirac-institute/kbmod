import warnings

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("Unable to determine the package version. " "This is likely a broken installation.")

# This is needed for compatibility with some older compilers: erfinv needs to be
# imported before other packages though I'm not sure why.
from scipy.special import erfinv

from . import (
    analysis,
    data_interface,
    file_utils,
    filters,
    jointfit_functions,
    result_list,
    run_search,
)

from .search import PSF, RawImage, LayeredImage, ImageStack, StackSearch
from .standardizers import Standardizer, StandardizerConfig
from .image_collection import ImageCollection
