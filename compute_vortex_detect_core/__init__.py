"""
vortex core tracking computation

A module for tracking the instaneous vortex cores from CFD simulation cut-planes. 

Main Components:
- VortexDetect: Class to execute the full analysis workflow
"""

from .utils import print, setup_logging, init_logging_from_cut, timer
from .vortex_detect import VortexDetect