"""
Plot Invariant Core Module

This module provides functionality for plotting velocity invariants and extracting QR data
from computed velocity gradient tensor results and vortex detection results.

The module consists of:
- plot_invariant.py: Main orchestrator class
- data_loader.py: Data loading from HDF5 files
- combined_plotter.py: Combined QR plots across locations
- single_extract.py: Single location QR extraction
- visualizer.py: Global invariant and profile plots
- grid_maker.py: Grid interpolation utilities
- vortex_detector.py: Vortex detection utilities
- utils.py: Utility functions and logging

Usage:
    python -m plot_invariant_core --cut PIV1 --data-type LES --plot-all
"""

from .plot_invariant import PlotInvariant, parse_arguments, main
from .data_loader import (
    load_velocity_invariants, 
    load_connectivity, 
    load_combined_core_data
)
from .combined_plotter import create_combined_qr_plots
from .single_extract import extract_single_qr_data
from .visualizer import plot_global_invariants, plot_vortex_profiles
from .grid_maker import make_grid
from .vortex_detector import vortex, find_squares
from .utils import setup_rcparams, timer, print, init_logging_from_cut

__version__ = "1.0.0"
__author__ = "Patrick Guang Chen Deng"
__email__ = "patrickgc.deng@mail.utoronto.ca"
__github__ = 'https://github.com/pattydeng8760'
__linkedin__ = 'www.linkedin.com/in/patrick-gc-deng'
__institution__ = "University of Toronto Institute for Aerospace Studies (UTIAS)"

__all__ = [
    'PlotInvariant',
    'parse_arguments', 
    'main',
    'load_velocity_invariants',
    'load_connectivity',
    'load_combined_core_data',
    'create_combined_qr_plots',
    'extract_single_qr_data',
    'plot_global_invariants',
    'plot_vortex_profiles',
    'make_grid',
    'vortex',
    'find_squares',
    'setup_rcparams',
    'timer',
    'print',
    'init_logging_from_cut'
]