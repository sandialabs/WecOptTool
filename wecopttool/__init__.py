from importlib.metadata import metadata as get_metadata
metadata = get_metadata('wecopttool')
__version__ = metadata['Version']
__version_info__ = tuple(__version__.split('.'))
__title__ = metadata['Name']
__description__ = metadata['Summary']
__author__ = metadata['Author']
__uri__ = metadata['Home-page']
__license__ = metadata['License']
__doc__ = (f"{__description__} ({__title__}) developed by {__author__}." +
           f" See: {__uri__}.")

from wecopttool.core import *
from wecopttool import waves
from wecopttool import hydrostatics
from wecopttool import pto
from wecopttool import geom
