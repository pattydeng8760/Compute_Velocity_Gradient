#!/usr/bin/env python
import sys
from argparse import Namespace

# make sure this path points to where your module lives locally!
sys.path.insert(0, "./Compute_Velocity_Gradient")

from compute_velocity_gradient_core.velocity_invariant import main
from compute_velocity_gradient_core.utils import init_logging_from_cut

config = {
    "cut"        : "PIV2",
    #"parent_dir" : "/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_5AOA_LES/Isosurface",
    'parent_dir' : "/scratch/denggua1/Bombardier_LES/PIV_Data",
    #'parent_dir' : "/Volumes/LES Data/B_10AOA_LES/PostProc",
    #"parent_dir" : '/Volumes/LES Data/PIV_Data',
    #'parent_dir' : '/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/Isosurface/Extract_Cutplane_Fine',
    "output_dir" : ".",
    "nproc"      : 192,
    "reload"     : False,
    "nblocks"    : 800, 
    "data_type"  : "PIV",
    "velocity"   : 50,
    "angle_of_attack": 10,
    "limited_gradient": True
}

if __name__ == "__main__":
    args = Namespace(**config)
    main(args)