from __future__ import absolute_import

__version__ = "0.0.1-alpha.3"

from .data import *
from .augmentations import *
import logging

try:
    from .transforms import *
except ImportError:
    logging.warning("Could not import transforms. It seems that Pytorch is not installed.")