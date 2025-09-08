"""
Compute Vortex Plot Module

This module provides tools for extracting velocity invariants and generating QR plots
from CFD and PIV data. It modularizes the functionality from Single_QR_Extract.py
with the same structure as compute_vortex_detect_core.

Main Components:
- VortexPlot: Main orchestrator class
- data_loader: Load velocity invariants and connectivity data
- grid_maker: Grid interpolation utilities
- vortex_detector: Vortex detection algorithms
- invariant_extractor: Velocity invariant extraction
- plotter: Plot generation utilities
- data_saver: Save extracted data to HDF5
- utils: Logging and utility functions

Usage:
    python -m compute_vortex_plot --cut PIV1 --data-type LES
"""
import sys
sys.path.insert(0, "/project/p/plavoie/denggua1/Coordinates")
from .vortex_plot import VortexPlot, parse_arguments
from .data_loader import load_velocity_invariants, load_connectivity
from .utils import init_logging_from_cut, timer, print

# Version and metadata
__version__ = "1.0.0"
__author__ = "Patrick Guang Chen Deng"
__email__ = "patrickgc.deng@mail.utoronto.ca"
__github__ = 'https://github.com/pattydeng8760'
__linkedin__ = 'www.linkedin.com/in/patrick-gc-deng'
__institution__ = "University of Toronto Institute for Aerospace Studies (UTIAS)"
__description__ = "Vortex core plotting for QR analysis"

__all__ = ["VortexPlot", "parse_arguments", "init_logging_from_cut", "timer", "print"]