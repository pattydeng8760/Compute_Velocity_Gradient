import os
import sys
import argparse
import time
import numpy as np
import h5py
from .utils import timer, print, init_logging_from_cut
from .data_loader import load_velocity_invariants, load_connectivity
from .combined_plotter import create_combined_qr_plots
from .single_extract import extract_single_qr_data
from .visualizer import plot_global_invariants, plot_vortex_profiles

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate velocity invariant plots and extract QR data from computed results."
    )
    
    parser.add_argument(
        "--cut", "-c",
        type=str, required=True,
        help="Cutplane identifier (e.g., 'PIV1'). Required for single-location plots. For combined plots, used only for output organization."
    )
    
    parser.add_argument(
        "--data-type", "-d",
        type=str,
        default="LES",
        choices=["LES", "PIV"],
        help="Type of data to process. Defaults to 'LES'. Can be 'LES' or 'PIV'."
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./",
        help="Directory to save output files."
    )
    
    parser.add_argument(
        "--chord", "-ch",
        type=float,
        default=0.305,
        help="Chord length for dimensionless scaling (default: 0.305 m)."
    )
    
    parser.add_argument(
        "--velocity", "-v",
        type=float,
        default=30.0,
        help="Freestream velocity for dimensionless scaling (default: 30.0 m/s)."
    )
    
    parser.add_argument(
        "--angle-of-attack", "-a",
        type=float,
        default=10.0,
        help="Angle of attack in degrees (default: 10.0°)."
    )
    
    parser.add_argument(
        "--plot-combined",
        action="store_true",
        help="Generate combined QR plots across all vortex locations. Requires combined data file from multiple Single_QR_Extract runs."
    )
    
    parser.add_argument(
        "--plot-single",
        action="store_true",
        help="Generate single location QR extraction plots for the specified --cut."
    )
    
    parser.add_argument(
        "--plot-global",
        action="store_true",
        help="Generate global invariant plots."
    )
    
    parser.add_argument(
        "--plot-profiles",
        action="store_true",
        help="Generate vortex profile plots along PCA axes."
    )
    
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Generate all available plots."
    )
    
    return parser.parse_args()


class PlotInvariant:
    """Main class for plotting velocity invariants and extracting QR data."""
    
    def __init__(self, args):
        self.cut = args.cut
        self.data_type = args.data_type
        self.output_dir = args.output_dir
        self.chord = args.chord
        self.velocity = args.velocity
        self.angle_of_attack = args.angle_of_attack
        self.plot_combined = args.plot_combined or args.plot_all
        self.plot_single = args.plot_single or args.plot_all
        self.plot_global = args.plot_global or args.plot_all
        self.plot_profiles = args.plot_profiles or args.plot_all
        
        # Set up paths
        self.velocity_invariant_dir = os.path.join(
            self.output_dir, f"Velocity_Invariants_{self.cut}_{self.data_type}"
        )
        self.vortex_detect_dir = os.path.join(
            self.output_dir, f"Vortex_Detect_Results_{self.cut}_{self.data_type}"
        )
        self.qr_plots_dir = os.path.join(self.output_dir, "QR_Plots")
        
        # Create output directories
        os.makedirs(self.qr_plots_dir, exist_ok=True)
        
        # Validate prerequisites
        self._validate_prerequisites()
    
    def _validate_prerequisites(self):
        """Validate that required input files exist based on selected plot types."""
        missing_files = []
        
        # For combined plots, check combined data file
        if self.plot_combined:
            combined_data_file = os.path.join(self.output_dir, "Velocity_Invariants_Core_B_10AOA_LES_U30.h5")
            if not os.path.exists(combined_data_file):
                missing_files.append(combined_data_file)
        
        # For single location plots, check individual cut files
        if self.plot_single or self.plot_global or self.plot_profiles:
            required_files = [
                os.path.join(self.velocity_invariant_dir, f"Velocity_Invariants_{self.cut}.h5"),
                os.path.join(self.velocity_invariant_dir, f"Velocity_Invariants_{self.cut}_Mean.h5")
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required input files. Please run velocity gradient analysis "
                f"and vortex detection first. Missing files:\n" + 
                "\n".join(missing_files)
            )
    
    def __str__(self):
        return (
            f"PlotInvariant(cut={self.cut}, data_type={self.data_type}, "
            f"output_dir={self.output_dir}, chord={self.chord}, "
            f"velocity={self.velocity}, angle_of_attack={self.angle_of_attack})"
        )
    
    @timer
    def run(self):
        """Run the complete plotting and extraction pipeline."""
        print(f"\n{'Performing velocity invariant plotting and QR analysis.':=^100}\n")
        
        # Print setup information
        print('\n----> Input Parameters:')
        print(f'    Cut: {self.cut}')
        print(f'    Data type: {self.data_type}')
        print(f'    Velocity invariant directory: {self.velocity_invariant_dir}')
        print(f'    Vortex detection directory: {self.vortex_detect_dir}')
        print(f'    Output directory: {self.qr_plots_dir}')
        print(f'    Chord length: {self.chord} m')
        print(f'    Velocity: {self.velocity} m/s')
        print(f'    Angle of attack: {self.angle_of_attack}°')
        print(f'    Plot combined: {self.plot_combined}')
        print(f'    Plot single: {self.plot_single}')
        print(f'    Plot global: {self.plot_global}')
        print(f'    Plot profiles: {self.plot_profiles}')
        
        # Generate plots based on user selection
        if self.plot_combined:
            print('\n----> Generating combined QR plots across all locations...')
            self._generate_combined_plots()
        
        # Load single-location data only if needed for single-location analyses
        data = None
        connectivity = None
        if self.plot_single or self.plot_global or self.plot_profiles:
            print('\n----> Loading velocity invariant data for single location analysis...')
            data = load_velocity_invariants(self.cut, self.data_type, self.velocity_invariant_dir)
            connectivity = load_connectivity(self.cut, self.data_type, self.velocity_invariant_dir)
        
        if self.plot_single:
            print('\n----> Generating single location QR extraction and analysis...')
            self._generate_single_plots(data, connectivity)
        
        if self.plot_global:
            print('\n----> Generating global invariant visualization...')
            self._generate_global_plots(data, connectivity)
        
        if self.plot_profiles:
            print('\n----> Generating vortex profile analysis along PCA axes...')
            self._generate_profile_plots(data, connectivity)
        
        print(f"\n{'Velocity invariant plotting and QR analysis complete.':=^100}\n")
    
    def _generate_combined_plots(self):
        """Generate combined QR plots across all vortex locations."""
        try:
            # Load combined vortex core data
            data_file = os.path.join(self.output_dir, "Velocity_Invariants_Core_B_10AOA_LES_U30.h5")
            if os.path.exists(data_file):
                print(f'    Reading combined data from: {data_file}')
                create_combined_qr_plots(data_file, self.qr_plots_dir, self.chord)
                print(f'    Successfully generated combined plots in: {self.qr_plots_dir}')
            else:
                print(f"    Warning: Combined core data file not found: {data_file}")
                print(f"    Skipping combined plots generation.")
                print(f"    Run single location analysis first to populate the combined data file.")
        except Exception as e:
            print(f"    Error generating combined plots: {e}")
    
    def _generate_single_plots(self, data, connectivity):
        """Generate single location QR extraction plots."""
        try:
            print(f'    Processing single location analysis for cut: {self.cut}')
            extract_single_qr_data(
                data, connectivity, self.cut, self.data_type, 
                self.vortex_detect_dir, self.velocity_invariant_dir,
                self.chord, self.velocity, self.angle_of_attack
            )
            print(f'    Successfully completed single location analysis')
        except Exception as e:
            print(f"    Error generating single location plots: {e}")
    
    def _generate_global_plots(self, data, connectivity):
        """Generate global invariant plots."""
        try:
            print(f'    Processing global invariant plots for cut: {self.cut}')
            plot_global_invariants(
                data, connectivity, self.cut, self.data_type,
                self.vortex_detect_dir, self.velocity_invariant_dir,
                self.chord, self.velocity
            )
            print(f'    Successfully generated global invariant plots')
        except Exception as e:
            print(f"    Error generating global plots: {e}")
    
    def _generate_profile_plots(self, data, connectivity):
        """Generate vortex profile plots along PCA axes."""
        try:
            print(f'    Processing vortex profile plots for cut: {self.cut}')
            plot_vortex_profiles(
                data, connectivity, self.cut, self.data_type,
                self.vortex_detect_dir, self.velocity_invariant_dir,
                self.chord, self.velocity
            )
            print(f'    Successfully generated vortex profile plots')
        except Exception as e:
            print(f"    Error generating profile plots: {e}")


def main(args=None):
    """Main function for plotting velocity invariants."""
    # Parse CLI args
    if args is None:
        args = parse_arguments()
    
    # Redirect stdout to log file and set up logging
    init_logging_from_cut(args.cut, args.data_type)
    
    # Build and run
    runner = PlotInvariant(args)
    runner.run()


if __name__ == "__main__":
    main()