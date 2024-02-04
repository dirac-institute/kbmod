import warnings

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("Unable to determine the package version. This is likely a broken installation.")


import logging
import time

logging.Formatter.converter = time.gmtime
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s %(levelname)s %(name)s] %(message)s')

from . import (
    analysis,
    analysis_utils,
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
