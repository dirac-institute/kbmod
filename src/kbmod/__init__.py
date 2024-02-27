import os
import time
import logging as _logging
from .search import Logging

import warnings

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("Unable to determine the package version. This is likely a broken installation.")


# there are ways for this to go to a file, but it's not worth it, short as it is
# The timezone converter can not be configured via the config submodule for
# some reason, only directly.
LOGGING_CONFIG = {
    "level": os.environ.get("KB_LOG_LEVEL", "WARNING"),
    "format": "[%(asctime)s %(levelname)s %(name)s] %(message)s",
    "datefmt": "%Y-%m-%dT%H:%M:%SZ",
}

_logging.Formatter.converter = time.gmtime
_logging.basicConfig(**LOGGING_CONFIG)

# duplicate the configuration on the CPP side, but let's try not to trample or
# hard-code any resolution of the additional required params that might haopen
# Python's side of the fence
__log_config = LOGGING_CONFIG
__log_config["converter"] = _logging.Formatter.converter.__name__
Logging().setConfig(__log_config)

# Import the rest of the package
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
