"""
velocity_invariant_computation

A module for computing velocity gradient invariants (P, Q, R),
strain-rate (Qs, Rs), and rotation (Qw) tensors from CFD simulation cut-planes.

Main Components:
- VelocityInvariant: Class to execute the full analysis workflow
"""

from .utils import print, timer, setup_logging, init_logging_from_cut
from .extractor import extract_gradient
from .invariants_pqr import compute_PQR_vectorized
from .invariants_sqw import compute_SQW_vectorized
from .writer import save_output_main, save_output_strain
from .velocity_invariant import VelocityInvariant, parse_arguments