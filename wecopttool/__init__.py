"""
WEC Design Optimization Toolbox (*WecOptTool*) developed by
Sandia National Laboratories. See
`snl-waterpower.github.io/WecOptTool/ <https://snl-waterpower.github.io/WecOptTool/>`_.

The top-level :python:`wecopttool` module contains:

* The :python:`wecopttool.WEC` class, which is the main way to interact
  with *WecOptTool*.
* Support functions for basic functionality, accessed as
  :python:`wecoptool.<function>`.

Other functionalities are implemented in the submodules, and can be
accessed as :python:`wecopttool.<module>.<function>`.



**Type Aliases**

+-------------------------+----------------------------------------------------------------------------+
| Alias                   | Type                                                                       |
+=========================+============================================================================+
| :python:`StateFunction` | :python:`Callable[[WEC, np.ndarray, np.ndarray, xr.Dataset], np.ndarray]`  |
+-------------------------+----------------------------------------------------------------------------+
"""


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

# logging
_handler = logging.StreamHandler()
_formatter = logging.Formatter(logging.BASIC_FORMAT)
_handler.setFormatter(_formatter)

_log = logging.getLogger(__name__)
_log.addHandler(_handler)

_log_capytaine = logging.getLogger("capytaine")
_log_capytaine.addHandler(_handler)


def set_loglevel(level:str) -> None:
    """ Change the logging level of the :python:`wecopttool` and
    :python:`capytaine` loggers to the specified level.

    Parameters
    ----------
    level
        Level for :py:meth:`python.logging.Logger.setLevel`.
        See `list of logging levels <https://docs.python.org/3/library/logging.html#levels>`_.
    """
    _log.setLevel(level.upper())
    _log_capytaine.setLevel(level.upper())
