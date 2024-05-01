import warnings

try:
    from ._version import version as __version__  # noqa: F401
except ImportError:
    warnings.warn("Unable to determine the package version. " "This is likely a broken installation.")

import os
import time
import logging as _logging
from logging import config as _config

# Import the rest of the package
from kbmod.search import Logging

# there are ways for this to go to a file, but is it worth it?
# Then we have to roll a whole logging.config_from_shared_config thing
_SHARED_LOGGING_CONFIG = {
    "level": os.environ.get("KB_LOG_LEVEL", "WARNING"),
    "format": "[%(asctime)s %(levelname)s %(name)s] %(message)s",
    "datefmt": "%Y-%m-%dT%H:%M:%SZ",
    "converter": "gmtime",
}

# Declare our own root logger, so that we don't start printing DEBUG
# messages from every package we import
__PY_LOGGING_CONFIG = {
    "version": 1.0,
    "formatters": {
        "standard": {
            "format": _SHARED_LOGGING_CONFIG["format"],
        },
    },
    "handlers": {
        "default": {
            "level": _SHARED_LOGGING_CONFIG["level"],
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        }
    },
    "loggers": {
        "kbmod.search.psi_phi_array": {"handlers": ["default"]},
        "kbmod.search.run_search": {"handlers": ["default"]},
        "kbmod.search.stamp_creator": {"handlers": ["default"]},
        "kbmod.search.trajectory_list": {"handlers": ["default"]},
    },
}

# The timezone converter can not be configured via the config submodule for
# some reason, only directly. Must be configured after loading the dictConfig
_config.dictConfig(__PY_LOGGING_CONFIG)
if _SHARED_LOGGING_CONFIG["converter"] == "gmtime":
    _logging.Formatter.converter = time.gmtime
else:
    _logging.Formatter.converter = time.localtime


# Some loggers are C++ loggers that have no equivalents, no natural module
# where they could be declared, in Python because f.e. the module is a fully
# pybind11 wrap. So they are pre-registered with the logging via the config.
# Register anything remotely to do with KBMOD with the C++ Logging singleton
# to ensure dispatching the log calls from C++ finds the appropriate logger.
for name in _logging.root.manager.loggerDict:
    Logging.registerLogger(_logging.getLogger(name))


from . import (  # noqa: F401
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
