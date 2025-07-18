#!/usr/bin/env python
"""
Wrapper script to run the compute_vortex_plot module from outside.
This script provides a convenient way to run vortex plot analysis with predefined configurations.
"""
import sys
from argparse import Namespace

# Add the current directory to Python path to ensure module can be imported
sys.path.insert(0, ".")
# Add window bounds to sys.path
sys.path.insert(0, "/project/p/plavoie/denggua1/Coordinates")

from compute_vortex_plot.vortex_plot import VortexPlot
from compute_vortex_plot.utils import init_logging_from_cut

# Configuration dictionary with default parameters
config = {
    "cut"              : "PIV1",                # Cutplane identifier
    "data_type"        : "LES",                 # Data type: 'LES' or 'PIV'
    "chord"            : 0.305,                 # Chord length for normalization
    "velocity"         : 30,                    # Free stream velocity
    "angle_of_attack"  : 10,                    # Angle of attack in degrees
    "grid_size"        : 500,                   # Grid size for interpolation
    "pca_points"       : 100,                   # Number of PCA query points
    "pca_length"       : 0.012,                 # PCA line length
    "limited_gradient" : True,                # Use limited gradient
}

def main():
    """Main function to run the vortex plot analysis."""
    print("="*80)
    print("Running Vortex Plot Analysis")
    print("="*80)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key:18}: {value}")
    print("="*80)
    
    # Create namespace from configuration
    args = Namespace(**config)
    
    # Initialize logging
    init_logging_from_cut(args.cut, args.data_type)
    
    # Create and run VortexPlot instance
    vp = VortexPlot(args)
    vp.run()

if __name__ == "__main__":
    main()