
from importlib.metadata import metadata as metadata
import logging

from wecopttool.core import *
from wecopttool import waves
from wecopttool import hydrostatics
from wecopttool import pto
from wecopttool import geom


# metadata
_metadata = metadata('wecopttool')
__version__ = _metadata['Version']
__version_info__ = tuple(__version__.split('.'))
__title__ = _metadata['Name']
__description__ = _metadata['Summary']
__author__ = _metadata['Author']
__uri__ = _metadata['Project-URL'].split(',')[1].strip()
__license__ = _metadata['License']
__doc__ = (f"{__description__} ({__title__}) developed by {__author__}." +
           f" See: {__uri__}.")

# logging
_handler = logging.StreamHandler()
_formatter = logging.Formatter(logging.BASIC_FORMAT)
_handler.setFormatter(_formatter)

_log = logging.getLogger(__name__)
_log.addHandler(_handler)

_log_capytaine = logging.getLogger("capytaine")
_log_capytaine.addHandler(_handler)


def set_loglevel(level):
    """ Change the logging level of the `wecopttool` and `capytaine`
    loggers to the specified level.
    """
    # TODO: include in API documentation
    _log.setLevel(level.upper())
    _log_capytaine.setLevel(level.upper())
