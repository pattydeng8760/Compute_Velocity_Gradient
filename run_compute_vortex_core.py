#!/usr/bin/env python
import sys
from argparse import Namespace

# make sure this path points to where your module lives locally!
sys.path.insert(0, "./Compute_Velocity_Gradient")

# Add window bounds to sys.path
sys.path.insert(0, "/project/rrg-plavoie/denggua1/Coordinates")

from compute_vortex_detect_core.vortex_detect import main
from compute_vortex_detect_core.utils import init_logging_from_cut

config = {
    "data_type"      : "LES",
    "cut"            : "095_TE",
    "parent_dir"     : "/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/Isosurface/Extract_Cutplane_Fine",
    "output_dir"     : ".",
    "nproc"          : 80,
    "method"         : "precise",
    "max_file"       : None,
    "angle_of_attack": 10,
    "plot"           : True,
    "velocity"       : 50,  # freestream velocity in m/s
    "chord"          : 0.3048,
    "plot_only"      : False  # Set to True to only generate plots from existing data
}

if __name__ == "__main__":
    args = Namespace(**config)
    main(args)
