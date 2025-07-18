import os, glob
import numpy as np
from multiprocessing import Pool, cpu_count
from .utils import timer, print, init_logging_from_cut
from .extractor import extract_gradient
from .invariants_pqr import compute_PQR_vectorized
from .invariants_sqw import compute_SQW_vectorized
from .writer import save_output_main, save_output_strain
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute velocity gradient invariants from CFD cutplanes."
    )
    parser.add_argument(
        "--cut", "-c",
        type=str, required=True,
        help="Cutplane identifier (e.g., '030_TE'). Directory is Cut_<cut>_VGT/ under parent_dir."
    )
    parser.add_argument(
        "--parent-dir", "-p",
        type=str,
        default="/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface",
        help="Top-level directory containing Cut_<cut>_VGT folders."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Directory to save output files."
    )
    parser.add_argument(
        "--nproc", "-n",
        type=int,
        default=4,
        help="Max number of processes to use (will be capped at available CPUs)."
    )
    parser.add_argument(
        "--reload", "-r",
        action="store_true",
        help="Force re-extraction of VGT even if cached file exists."
    )
    parser.add_argument(
        "--nblocks", "-b",
        type=int,
        default=1000,
        help="How many blocks to split the nodes into for parallelism."
    )
    parser.add_argument(
        "--data-type", "-d",
        type=str,
        default="LES",
        choices=["LES", "PIV"],
        help="Type of data to extract the velocity gradient tensor from. Defaults to 'LES'. Can be 'LES' or 'PIV'."
    )
    parser.add_argument(
        "--velocity", "-v",
        type=int, default = 30,
        help="The freestream velocity of the data."
    )
    parser.add_argument(
        "--angle-of-attack", "-aoa",
        type=float, default=10.0,
        help="Angle of attack of the data in degrees."
    )
    parser.add_argument(
        "--limited-gradient", "-lg",
        action="store_true",
        help="Use limited gradient computation for LES data (only applies to LES data type). Computes with limited VGT tensor corresponding to stereo-PIV availability where dv/dx and dw/dx are unavailable, and du/dx is calculated from incompressible assumption."
    )
    return parser.parse_args()


class VelocityInvariant:
    def __init__(self, args):
        self.cut        = args.cut
        self.reload     = args.reload
        self.parent_dir = args.parent_dir
        self.nproc      = args.nproc
        self.nblocks    = args.nblocks
        self.data_type = args.data_type
        self.velocity    = args.velocity
        self.angle_of_attack = args.angle_of_attack
        self.limited_gradient = args.limited_gradient
        
        # Update output directory name to include _limited suffix if flag is set
        output_suffix = f"_{self.data_type}"
        if self.limited_gradient and self.data_type == 'LES':
            output_suffix += "_Limited"
        self.output  = os.path.join(args.output_dir,f"Velocity_Invariants_{self.cut}{output_suffix}")
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)
        if self.data_type == 'LES':
            pattern = os.path.join(self.parent_dir, f"Cut_{self.cut}_VGT", "*.h5")
        elif self.data_type == 'PIV':
            cut_map = {
            "10": {"PIV1": "058","PIV2": "078","PIV3": "105"},
            "5": {"PIV1": "060","PIV2": "080"}}
            mapped_cut = cut_map[str(self.angle_of_attack)][self.cut]
            pattern = os.path.join(self.parent_dir, f"AoA{str(int(self.angle_of_attack))}_xc_{mapped_cut}",f"U{str(int(self.velocity ))}","PIV_Data","*.h5")
        self.arr     = sorted(glob.glob(pattern))
        if not self.arr:
            raise FileNotFoundError(f"No files found matching: {pattern}")
        os.makedirs(self.output, exist_ok=True)
    
    def __str__(self):
        return (
            f"VelocityInvariant(cut={self.cut}, reload={self.reload}, "
            f"parent_dir={self.parent_dir}, nproc={self.nproc}, "
            f"nblocks={self.nblocks}, output_dir={self.output},"
            f"data_type={self.data_type}, limited_gradient={self.limited_gradient})"
        )
        
    def extract_velocity_gradient(self):
        # Extract velocity gradient tensor and velocity from the cut
        velocity_gradient, velocity = extract_gradient(
            self.arr, self.cut, self.reload, self.output, time=None
        )
        node_count, time_steps = velocity_gradient.shape[2], velocity_gradient.shape[3]
        # etermine processes
        avail = cpu_count()
        nproc = min(avail, self.nproc)
        # split nodes into blocks
        node_indices = np.array_split(np.arange(node_count), self.nblocks)
        print('\n---->Partitioning data into {0:d} blocks.'.format(len(node_indices)))
        print(f'Number of available parallel compute processes: {nproc}')
        print(f'Number of nodes: {node_count}')
        print(f'Number of time steps: {time_steps}')
        blocks = [(velocity_gradient[:, :, indices, :], block_num) for block_num, indices in enumerate(node_indices)]
        
        return velocity, node_count, node_indices, time_steps, blocks

    @timer
    def run(self):
        print(f"\n{'Performing velocity gradient invariants computation.':=^100}\n")
        
        # Print setup information
        print('\n----> Input Parameters:')
        print(f'    Cut: {self.cut}')
        print(f'    Data type: {self.data_type}')
        print(f'    Parent directory: {self.parent_dir}')
        print(f'    Output directory: {self.output}')
        print(f'    Number of processes: {self.nproc}')
        print(f'    Number of blocks: {self.nblocks}')
        print(f'    Reload: {self.reload}')
        print(f'    Velocity: {self.velocity}')
        print(f'    Angle of attack: {self.angle_of_attack}')
        print(f'    Limited gradient: {self.limited_gradient}')
        print(f'    Total files found: {len(self.arr)}')
        
        # 1) extract the velocity gradient tensor and velocity from the cut
        velocity_gradient, velocity = extract_gradient(
            self.arr, self.cut, self.reload, self.output, time=None, data_type=self.data_type,
            limited_gradient=self.limited_gradient
        )
        node_count, time_steps = velocity_gradient.shape[2], velocity_gradient.shape[3]
        
        # 2) determine processes
        avail = cpu_count()
        nproc = min(avail, self.nproc)
        
        # 3) split nodes into blocks
        node_indices = np.array_split(np.arange(node_count), self.nblocks)
        print('\n----> Partitioning data into {0:d} blocks.'.format(len(node_indices)))
        print(f'    Number of available parallel compute processes: {nproc}')
        print(f'    Number of nodes: {node_count}')
        print(f'    Number of time steps: {time_steps}')
        blocks = [(velocity_gradient[:, :, indices, :], block_num) for block_num, indices in enumerate(node_indices)]
        del velocity_gradient  # free memory
        print('\n----> Perofrming Parallel VGT Invariant Calculations...')
        
        # 4) compute PQR
        with Pool(nproc) as pool:
            pqr_results = pool.starmap(compute_PQR_vectorized, blocks)
        save_output_main(
            node_count, node_indices, time_steps,
            pqr_results, self.arr, self.output, velocity, self.cut, self.limited_gradient
        )
        del pqr_results
        
        print('\n----> Perofrming Parallel Strain-Rotation Invariant Calculations...')
        # 5) compute Qs, Rs, Qw
        with Pool(nproc) as pool:
            sqw_results = pool.starmap(compute_SQW_vectorized, blocks)
        save_output_strain(
            node_count, node_indices, time_steps,
            sqw_results, self.arr, self.output, velocity, self.cut, self.limited_gradient
        )
        print(f"\n{'Velocity gradient invariants computation complete.':=^100}\n")

def main(args=None):
    # Parse CLI args
    if args is None:
        args = parse_arguments()
    # Redirect stdout to log_<cut>.txt and set up logging
    init_logging_from_cut(args.cut,args.data_type)
    # Build and run
    runner = VelocityInvariant(args)
    runner.run()

if __name__ == "__main__":
    main()