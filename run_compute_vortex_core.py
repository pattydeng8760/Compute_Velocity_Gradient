#!/usr/bin/env python
import sys
from argparse import Namespace

# make sure this path points to where your module lives locally!
sys.path.insert(0, "./Compute_Velocity_Gradient")

from compute_vortex_detect_core.vortex_detect import main
from compute_vortex_detect_core.utils import init_logging_from_cut

config = {
    "data_type"      : "LES",
    "cut"            : "PIV1",
    "parent_dir"     : "/Volumes/LES Data/B_10AOA_LES/PostProc",
    "output_dir"     : ".",
    "nproc"          : 16,
    "method"         : "precise",
    "max_file"       : 16,
    "angle_of_attack": 10,
    "plot"           : True,
    "chord"          : 0.3048
}

if __name__ == "__main__":
    args = Namespace(**config)
    main(args)
