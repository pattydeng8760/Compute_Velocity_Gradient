"""
Vortex Core Detection and Tracking Module

A comprehensive module for detecting and tracking instantaneous vortex cores from CFD simulation 
and PIV experimental cut-planes. The module provides parallel processing capabilities for analyzing
large datasets and extracting vortex characteristics including core locations, wandering patterns,
and statistical distributions.

Main Components:
- VortexDetect: Main class to execute the complete vortex detection workflow
- vortex: Class representing individual vortex objects with core detection methods
- vortex_trace: Class for computing vortex wandering statistics
- make_grid: Class for interpolating unstructured data to structured grids
- Plotting utilities: Advanced visualization functions with PCA analysis

Key Features:
- Parallel processing for large datasets
- Support for LES and PIV data types
- Multiple vortex detection methods (max, precise, area)
- Advanced visualization with PCA trajectory analysis
- Statistical analysis of vortex wandering
- Professional publication-quality plots

Usage:
    from compute_vortex_detect_core import VortexDetect
    
    # Command line usage
    python -m compute_vortex_detect_core --cut PIV1 --data-type LES --plot
    
    # Programmatic usage
    from argparse import Namespace
    args = Namespace(cut='PIV1', data_type='LES', ...)
    detector = VortexDetect(args)
    results = detector.run()
"""

# Main classes and functions
from .vortex_detect import VortexDetect, parse_arguments, main
from .vortex_track import vortex, vortex_trace, find_squares
from .make_grid import make_grid
from .detect_vortex import detect_vortex, process_file_block
from .save_data import save_data
from .plot_results import plot_all_results, plot_vortex_cores, plot_probability_distribution
from .utils import print, init_logging_from_cut, timer

# Version and metadata
__version__ = "1.0.0"
__author__ = "Patrick Guang Chen Deng"
__email__ = "patrickgc.deng@mail.utoronto.ca"
__github__ = 'https://github.com/pattydeng8760'
__linkedin__ = 'www.linkedin.com/in/patrick-gc-deng'
__institution__ = "University of Toronto Institute for Aerospace Studies (UTIAS)"
__description__ = "Vortex core detection and tracking for CFD/PIV data"

# Public API
__all__ = [
    # Main classes
    'VortexDetect',
    'vortex',
    'vortex_trace',
    'make_grid',
    
    # Core functions
    'detect_vortex',
    'process_file_block',
    'parse_arguments',
    'main',
    
    # Data handling
    'save_data',
    'find_squares',
    
    # Plotting functions
    'plot_all_results',
    'plot_vortex_cores',
    'plot_probability_distribution',
    
    # Utilities
    'print',
    'init_logging_from_cut',
    'timer',
]