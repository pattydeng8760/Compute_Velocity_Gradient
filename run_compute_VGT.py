#!/usr/bin/env python
import sys
from argparse import Namespace

# make sure this path points to where your module lives locally!
sys.path.insert(0, "./Compute_Velocity_Gradient")

from compute_velocity_gradient_core.velocity_invariant import main
from compute_velocity_gradient_core.utils import init_logging_from_cut

config = {
    "cut"        : "PIV1",
    #"parent_dir" : "/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface",
    'parent_dir' : "/home/p/plavoie/denggua1/scratch/Bombardier_LES/PIV_Data",
    "output_dir" : ".",
    "nproc"      : 16,
    "reload"     : False,
    "nblocks"    : 1200, 
    "data_type"  : "PIV",
    "velocity"   : 30,
    "angle_of_attack": 10,  
}

if __name__ == "__main__":
    args = Namespace(**config)
    main(args)