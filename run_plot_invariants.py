#!/usr/bin/env python
import sys
from argparse import Namespace

# Make sure this path points to where your module lives locally!
sys.path.insert(0, "./Compute_Velocity_Gradient")

from plot_invariant_core.plot_invariant import main
from plot_invariant_core.utils import init_logging_from_cut

config = {
    "cut": "PIV2",
    "data_type": "LES",
    "output_dir": ".",
    "chord": 0.305,
    "velocity": 30.0,
    "angle_of_attack": 10.0,
    "plot_combined": True,
    "plot_single": True,
    "plot_global": True,
    "plot_profiles": True,
    "plot_all": False,  # Set to True to enable all plots
}

if __name__ == "__main__":
    args = Namespace(**config)
    main(args)