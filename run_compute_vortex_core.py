#!/usr/bin/env python
import sys
from argparse import Namespace

# make sure this path points to where your module lives locally!
sys.path.insert(0, "./Compute_Velocity_Gradient")

# Add window bounds to sys.path
sys.path.insert(0, "/project/p/plavoie/denggua1/Coordinates")

from compute_vortex_detect_core.vortex_detect import main
from compute_vortex_detect_core.utils import init_logging_from_cut

config = {
    "data_type"      : "PIV",
    "cut"            : "PIV3",
    "parent_dir"     : "/home/p/plavoie/denggua1/scratch/Bombardier_LES/PIV_Data",
    "output_dir"     : ".",
    "nproc"          : 20,
    "method"         : "precise",
    "max_file"       : None,
    "angle_of_attack": 10,
    "plot"           : True,
    "chord"          : 0.3048,
    "plot_only"      : False  # Set to True to only generate plots from existing data
}

if __name__ == "__main__":
    args = Namespace(**config)
    main(args)
