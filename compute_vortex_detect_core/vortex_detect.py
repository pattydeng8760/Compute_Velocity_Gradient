import os, glob
import numpy as np
import multiprocessing
import time
import argparse
from multiprocessing import Pool, cpu_count
from .utils import timer, print, init_logging_from_cut
from .detect_vortex import detect_vortex
from .plot_results import plot_all_results

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Detect vortices from CFD or PIV cutplanes using parallel processing."
    )
    
    parser.add_argument(
        "--data-type", "-d",
        type=str,
        default="LES",
        choices=["LES", "PIV"],
        help="Type of data to perform vortex core detectin on. Defaults to 'LES'. Can be 'LES' or 'PIV'."
    )
    
    parser.add_argument(
        "--cut", "-c",
        type=str, required=True,
        help="Cutplane identifier (e.g., 'PIV1'). Directory is Cut_<cut>_VGT/ under parent_dir."
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
        default="./",
        help="Directory to save output files."
    )
    parser.add_argument(
        "--nproc", "-n",
        type=int,
        default=None,
        help="Max number of processes to use (will be capped at available CPUs)."
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="precise",
        choices=["max", "precise", "area"],
        help="Method for vortex detection."
    )
    parser.add_argument(
        "--max-file",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)."
    )
    parser.add_argument(
        "--angle-of-attack", "-a",
        type=int,
        default=5,
        help="Angle of attack for window boundary selection."
    )
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Generate plots of vortex detection results."
    )
    
    parser.add_argument(
        "--chord", "-co",
        type=float,
        default=0.3048,
        help="Chord length for plotting (default is 0.3048 m). This is used in Plot_Result."
    )
    
    parser.add_argument(
        "--plot-only", 
        action="store_true",
        help="Only generate plots from existing saved data files. Skip vortex detection computation. "
             "Requires all data files (Grid_<cut>_Data.mat, *_core_<cut>.npy, *_core_<cut>_Diff.npy) to exist."
    )
    
    return parser.parse_args()

class VortexDetect:
    """Main class for handling vortex detection in parallel."""
    
    def __init__(self, args):
        self.data_type = args.data_type
        self.args = args
        self.cut = args.cut
        self.parent_dir = args.parent_dir
        self.output_dir = args.output_dir
        self.nproc = args.nproc
        self.method = args.method
        self.max_file = args.max_file
        self.alpha = args.angle_of_attack
        self.chord = args.chord
        self.plot = args.plot
        self.plot_only = args.plot_only
        
        # Set up directories
        self.source_dir = os.path.join(self.parent_dir, f'Cut_{self.cut}_VGT')
        self.sol_dir = f'Vortex_Detect_Results_{self.cut}_{self.data_type}'
        
        # Create output directory
        os.makedirs(self.sol_dir, exist_ok=True)
        
        # Set number of processes
        if self.nproc is None:
            self.nproc = multiprocessing.cpu_count()
        
    def run(self):
        """Run the complete vortex detection pipeline."""
        t = time.time()
        
        print(f"\n{'Performing vortex detection analysis.':=^100}\n")
        
        # Print setup information
        print('\n----> Input Parameters:')
        print(f'    Data type: {self.data_type}')
        print(f'    Cut: {self.cut}')
        print(f'    Parent directory: {self.parent_dir}')
        print(f'    Source directory: {self.source_dir}')
        print(f'    Output directory: {self.sol_dir}')
        print(f'    Number of processes: {self.nproc}')
        print(f'    Detection method: {self.method}')
        print(f'    Max files (testing): {self.max_file if self.max_file else "None"}')
        print(f'    Angle of attack: {self.alpha}')
        print(f'    Chord length: {self.chord}')
        print(f'    Generate plots: {self.plot}')
        print(f'    Plot only mode: {self.plot_only}')
        
        # Check if plot-only mode is requested
        if self.plot_only:
            print('\n----> Plot-only mode: Checking for existing data files...')
            if self.check_required_files():
                print('    All required files found. Generating plots only.')
                plot_all_results(self.cut, self.sol_dir, self.chord, self.data_type)
                S_core_loc, S_Vort_Diff, P_core_loc, P_Vort_Diff, T_core_loc, T_Vort_Diff, Vars = None, None, None, None, None, None, None
            else:
                print('    Error: Required data files not found. Cannot generate plots.')
                print('    Please run vortex detection first without --plot-only flag.')
                return None
        else:
            # Detect vortices
            S_core_loc, S_Vort_Diff, P_core_loc, P_Vort_Diff, T_core_loc, T_Vort_Diff, Vars = detect_vortex(
                self.source_dir, 
                self.cut, 
                self.alpha, 
                self.method, 
                nb_tasks=self.nproc, 
                max_file=self.max_file, 
                output_dir=self.sol_dir
            )
            
            # Plot results if requested
            if self.plot:
                plot_all_results(self.cut, self.sol_dir, self.chord, self.data_type)
        
        elapsed = time.time() - t
        print(f"\n{'Vortex detection analysis complete.':=^100}\n")
        print('\n----> Timing Information:')
        print(f'    Total calculation time: {elapsed:1.0f} s')
        
        return S_core_loc, S_Vort_Diff, P_core_loc, P_Vort_Diff, T_core_loc, T_Vort_Diff, Vars
    
    def check_required_files(self):
        """
        Check if all required data files exist for plotting.
        
        Returns:
            bool: True if all required files exist, False otherwise
        """
        print('    Checking for required data files:')
        
        # List of required files
        required_files = [
            f'Grid_{self.cut}_Data.mat',
            f'S_core_{self.cut}.npy',
            f'P_core_{self.cut}.npy',
            f'S_core_{self.cut}_Diff.npy',
            f'P_core_{self.cut}_Diff.npy'
        ]
        
        # Add mask file for non-PIV3 cuts
        if self.cut != 'PIV3':
            required_files.append(f'Grid_mask_index_{self.cut}.npy')
        
        # Check for tertiary vortex files (conditional)
        tertiary_cuts = ['PIV2', 'PIV3', '015_TE', '085_TE', '095_TE']
        if self.cut in tertiary_cuts:
            tertiary_files = [
                f'T_core_{self.cut}.npy',
                f'T_core_{self.cut}_Diff.npy'
            ]
            # Check if tertiary files exist (optional)
            for tf in tertiary_files:
                tf_path = os.path.join(self.sol_dir, tf)
                if os.path.exists(tf_path):
                    required_files.append(tf)
                    print(f'        Found optional tertiary file: {tf}')
        
        # Check each required file
        missing_files = []
        for filename in required_files:
            filepath = os.path.join(self.sol_dir, filename)
            if os.path.exists(filepath):
                print(f'        ✓ Found: {filename}')
            else:
                print(f'        ✗ Missing: {filename}')
                missing_files.append(filename)
        
        if missing_files:
            print(f'    Missing {len(missing_files)} required file(s)')
            return False
        else:
            print(f'    All {len(required_files)} required files found')
            return True
    
    def plot_results(self):
        """
        Generate plots from existing saved data files.
        
        This method is a convenience function that can be called directly
        to generate plots without running the full detection pipeline.
        """
        print('\n----> Generating plots from existing data...')
        
        if self.check_required_files():
            plot_all_results(self.cut, self.sol_dir, self.chord, self.data_type)
            print('    Plot generation completed successfully')
        else:
            print('    Error: Cannot generate plots - required files missing')
            print('    Please run vortex detection first to generate the required data files')

def main(args=None):
    """Main function for handling vortex detection in parallel."""
    # Parse CLI args
    if args is None:
        args = parse_arguments()
    
    # Redirect stdout to log_<cut>.txt and set up logging
    init_logging_from_cut(args.cut, args.data_type)
    
    # Build and run
    runner = VortexDetect(args)
    return runner.run()