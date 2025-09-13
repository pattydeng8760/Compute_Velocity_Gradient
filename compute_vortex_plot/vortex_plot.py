import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from .utils import timer, print, init_logging_from_cut, _setup_plot_params
from .data_loader import load_velocity_invariants, load_connectivity
from .grid_maker import make_grid
from .vortex_detector import vortex, piv_vortex
from .invariant_extractor import extract_velocity_invariants
from .single_plot import plot_global_invariants, plot_vortex_profiles, plot_spectra
from .combine_spectra_plot import plot_combine_spectra
from .combine_qr_plot import plot_combine_qr
from .data_saver import save_extracted_data
from window_bounds import get_window_boundaries

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract velocity invariants and generate QR plots from CFD or PIV data."
    )
    
    parser.add_argument(
        "--data-type", "-d",
        type=str,
        default="LES",
        choices=["LES", "PIV"],
        help="Type of data to process. Defaults to 'LES'. Can be 'LES' or 'PIV'."
    )
    
    parser.add_argument(
        "--cut", "-c",
        type=str, required=True,
        help="Cutplane identifier (e.g., 'PIV1')."
    )
    
    parser.add_argument(
        "--chord", "-ch",
        type=float,
        default=0.305,
        help="Chord length for normalization (default: 0.305)."
    )
    
    parser.add_argument(
        "--velocity", "-v",
        type=int,
        default=30,
        help="Free stream velocity for normalization (default: 30)."
    )
    
    parser.add_argument(
        "--angle-of-attack", "-a",
        type=int,
        default=10,
        help="Angle of attack in degrees (default: 10)."
    )
    
    parser.add_argument(
        "--grid-size", "-g",
        type=int,
        default=500,
        help="Grid size for interpolation (default: 500)."
    )
    
    parser.add_argument(
        "--pca-points", "-p",
        type=int,
        default=100,
        help="Number of PCA query points (default: 100)."
    )
    
    parser.add_argument(
        "--pca-length", "-l",
        type=float,
        default=0.012,
        help="PCA line length (default: 0.012)."
    )
    
    parser.add_argument(
        "--limited-gradient", "-lg",
        action="store_true",
        help="Use limited gradient computation for LES data (only applies to LES data type). Computes with limited VGT tensor corresponding to stereo-PIV availability where dv/dx and dw/dx are unavailable, and du/dx is calculated from incompressible assumption."
    )
    
    parser.add_argument(
        "--plot-all", "-pa",
        action="store_true",
        help="Generate all plots including single-location plots and combined Q-R plots across locations."
    )
    
    parser.add_argument(
        "--locations", "-loc",
        type=str,
        nargs='+',
        default=None,
        help="List of locations for combined Q-R plotting (e.g., --locations 030_TE PIV1 PIV2 085_TE 095_TE PIV3). If not specified, default locations will be used based on data type."
    )
    
    parser.add_argument(
        "--plot-spectra", "-ps",
        type=float,
        default=0.186e-8*2000,
        help="Time plotting the spectra content at the core positions."
    )
        
    parser.add_argument(
        "--time-step", "-dt",
        type=float,
        default=0.186e-8*2000,
        help="Time step between frames in seconds for the full solution (default: 0.186e-8*2000)."
    )
    
    return parser.parse_args()

class VortexPlot:
    def __init__(self, args):
        self.args = args
        self.location = args.cut
        self.data_type = args.data_type
        self.chord = args.chord
        self.velocity = args.velocity
        self.angle_of_attack = args.angle_of_attack
        self.grid_size = args.grid_size
        self.pca_points = args.pca_points
        self.pca_length = args.pca_length
        self.limited_gradient = args.limited_gradient
        self.plot_all = args.plot_all
        self.plot_spectra = args.plot_spectra
        self.locations = args.locations
        self.dt = args.time_step
        # Set default locations based on data type if not specified
        if self.locations is None:
            if self.data_type.upper() == 'LES':
                self.locations = ['030_TE', 'PIV1', 'PIV2', '085_TE', '095_TE', 'PIV3']
            elif self.data_type.upper() == 'PIV':
                self.locations = ['PIV1', 'PIV2', 'PIV3']
            else:
                self.locations = ['030_TE', 'PIV1', 'PIV2', '085_TE', '095_TE', 'PIV3']
        
        # Create output directory
        self.output_dir = f'Velocity_Invariants_{self.location}_{self.data_type}'
        
        if self.limited_gradient and self.data_type == 'LES':
            self.output_dir += '_Limited'
            print("Using limited gradient computation for LES data, the output directory is set to:", self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        print('\n----> Loading velocity invariants data...')
        self.data = load_velocity_invariants(self.location, self.data_type, self.limited_gradient)
        print('----> Loading connectivity data...')
        self.connectivity = load_connectivity(self.location, self.data_type, self.limited_gradient)
        print('    Data loading complete.')
    
    def create_grid(self):
        print('\n----> Creating interpolation grid...')
        
        if self.data_type == 'PIV':
            # PIV data is already structured - no interpolation needed
            print('    PIV data detected - using structured grid directly')
            self.data_grid = self._create_piv_structured_grid()
        else:
            # LES data requires interpolation
            print('    LES data detected - performing interpolation')
            self.data_grid = self._create_les_interpolated_grid()
        
        print('    Grid creation complete.')
    
    def _create_piv_structured_grid(self):
        """Create PIV grid object directly from structured data without interpolation."""
        from .grid_maker import PIVStructuredGrid
        
        # PIV data structure analysis
        unique_y = np.unique(self.data['y'])
        unique_z = np.unique(self.data['z'])
        ny, nz = len(unique_y), len(unique_z)
        
        print(f'    PIV structured grid dimensions: {ny} x {nz}')
        print(f'    Y range: {unique_y.min():.6f} to {unique_y.max():.6f}')
        print(f'    Z range: {unique_z.min():.6f} to {unique_z.max():.6f}')
        
        # Create structured grid object
        data_grid = PIVStructuredGrid(
            self.data['y'], self.data['z'], 
            unique_y, unique_z,
            ny, nz
        )
        
        # Add mean variables directly without interpolation
        variables = [
            ('vort_x', np.mean(self.data['vort_x'], axis=1)),
            ('Phat', np.mean(self.data['Phat_all'], axis=1)),
            ('Qhat', np.mean(self.data['Qhat_all'], axis=1)),
            ('Rhat', np.mean(self.data['Rhat_all'], axis=1)),
            ('Pressure_Hessian', np.mean(self.data['pressure_hessian'], axis=1)),
            ('Pressure', np.mean(self.data['pressure'], axis=1)),
            ('Qs', np.mean(self.data['Qs_all'], axis=1)),
            ('Rs', np.mean(self.data['Rs_all'], axis=1)),
            ('Qw', np.mean(self.data['Qw_all'], axis=1)),
            ('var_A', self.data['var_A']),
            ('var_S', self.data['var_S']),
            ('var_omega', self.data['var_omega']),
            ('mean_SR', self.data['mean_SR'])
        ]
        
        for var_name, var_data in variables:
            data_grid.add_variable(var_name, var_data)
        
        return data_grid
    
    def _create_les_interpolated_grid(self):
        """Create LES grid object with interpolation."""
        from .grid_maker import make_grid
        
        # LES data bounds
        y_bnd = [-0.05, 0.05]
        z_bnd = [-0.16, -0.06]
        
        print(f'    Grid resolution: {self.grid_size} x {self.grid_size}')
        print(f'    Y bounds: {y_bnd}')
        print(f'    Z bounds: {z_bnd}')
        
        # Create grid for plotting global invariants
        data_grid = make_grid(
            self.grid_size, y_bnd, z_bnd, 
            self.data['y'], self.data['z'], 
            np.mean(self.data['u'], axis=1),
            np.mean(self.data['vort_x'], axis=1), 
            'vort_x'
        )
        
        # Calculate grid for all variables
        variables = [
            ('Phat', np.mean(self.data['Phat_all'], axis=1)),
            ('Qhat', np.mean(self.data['Qhat_all'], axis=1)),
            ('Rhat', np.mean(self.data['Rhat_all'], axis=1)),
            ('Pressure_Hessian', np.mean(self.data['pressure_hessian'], axis=1)),
            ('Pressure', np.mean(self.data['pressure'], axis=1)),
            ('Qs', np.mean(self.data['Qs_all'], axis=1)),
            ('Rs', np.mean(self.data['Rs_all'], axis=1)),
            ('Qw', np.mean(self.data['Qw_all'], axis=1)),
            ('var_A', self.data['var_A']),
            ('var_S', self.data['var_S']),
            ('var_omega', self.data['var_omega']),
            ('mean_SR', self.data['mean_SR'])
        ]
        
        for var_name, var_data in variables:
            data_grid.calculate_grid(self.grid_size, y_bnd, z_bnd, var_data, var_name)
        
        return data_grid
    
    def detect_vortices(self):
        print('\n----> Detecting vortices...')
        
        # Get window boundaries based on data type
        if self.data_type == 'PIV':
            from window_bounds import get_window_boundaries_PIV
            boundaries = get_window_boundaries_PIV(self.location, str(self.angle_of_attack))
            print(f'    PIV window boundaries loaded for location: {self.location}')
        else:
            boundaries = get_window_boundaries(self.location, str(self.angle_of_attack))
            print(f'    LES window boundaries loaded for location: {self.location}')
        
        mean_vort_x = np.mean(self.data['vort_x'], axis=1)
        mean_Q = np.mean(np.nan_to_num(self.data['Qhat_all'], nan=0.0), axis=1)
        mean_u = np.mean(self.data['u'], axis=1)
        
        # Adjust detection parameters for PIV data
        if self.data_type == 'PIV':
            # PIV data typically has different vorticity levels and velocity thresholds
            primary_level = -20  # Much less strict for PIV data
            secondary_level = -20
            tertiary_level = -20
            shear_level = -15 
            primary_method = 'max'  # Max method works better for PIV
            secondary_method = 'max'
        else:
            # LES data parameters
            primary_level = -35
            secondary_level = -35
            tertiary_level = -30
            shear_level = -15
            primary_method = 'precise'
            secondary_method = 'precise'
        
        # Choose vortex detector based on data type
        if self.data_type == 'PIV':
            vortex_detector = piv_vortex
        else:
            vortex_detector = vortex
        
        # Primary vortex
        print('    Detecting primary vortex...')
        self.P_Vortex = vortex_detector(
            'Primary', boundaries['PV_WindowLL'], boundaries['PV_WindowUR'],
            self.data['y'], self.data['z'], mean_u, 
            mean_vort_x * self.chord / self.velocity, 
            choice=primary_method, level=primary_level
        )
        print(f'        Primary vortex core: {self.P_Vortex.core.core_loc[0]}')
        
        # Secondary vortex
        print('    Detecting secondary vortex...')
        self.S_Vortex = vortex_detector(
            'Secondary', boundaries['SV_WindowLL'], boundaries['SV_WindowUR'],
            self.data['y'], self.data['z'], mean_u, 
            mean_vort_x * self.chord / self.velocity, 
            choice=secondary_method, level=secondary_level
        )
        print(f'        Secondary vortex core: {self.S_Vortex.core.core_loc[0]}')
        
        # Optional tertiary and shear vortices
        self.loc_points_aux = []
        aux_vortex_count = 0
        
        # Tertiary vortex
        TV_WindowLL = boundaries.get('TV_WindowLL')
        TV_WindowUR = boundaries.get('TV_WindowUR')
        if TV_WindowLL is not None and TV_WindowUR is not None and len(TV_WindowLL) == 2 and len(TV_WindowUR) == 2:
            print('    Detecting tertiary vortex...')
            self.T_Vortex = vortex_detector(
                'Tertiary', TV_WindowLL, TV_WindowUR,
                self.data['y'], self.data['z'], mean_u, 
                mean_vort_x * self.chord / self.velocity, 
                choice='area', level=tertiary_level
            )
            self.loc_points_aux.append(self.T_Vortex)
            aux_vortex_count += 1
            print(f'        Tertiary vortex core: {self.T_Vortex.core.core_loc[0]}')
        
        # Shear vortices
        SS_shear_windowLL = boundaries.get('SS_shear_windowLL')
        SS_shear_windowUR = boundaries.get('SS_shear_windowUR')
        if SS_shear_windowLL is not None and SS_shear_windowUR is not None and len(SS_shear_windowLL) == 2 and len(SS_shear_windowUR) == 2:
            print('    Detecting secondary shear vortex...')
            self.SS_shear = vortex_detector(
                'SS_shear', SS_shear_windowLL, SS_shear_windowUR,
                self.data['y'], self.data['z'], mean_u, 
                mean_vort_x * self.chord / self.velocity, 
                choice='area', level=shear_level
            )
            self.loc_points_aux.append(self.SS_shear)
            aux_vortex_count += 1
            print(f'        Secondary shear vortex core: {self.SS_shear.core.core_loc[0]}')
        
        PS_shear_windowLL = boundaries.get('PS_shear_windowLL')
        PS_shear_windowUR = boundaries.get('PS_shear_windowUR')  # Fixed typo
        if PS_shear_windowLL is not None and PS_shear_windowUR is not None and len(PS_shear_windowLL) == 2 and len(PS_shear_windowUR) == 2:
            print('    Detecting primary shear vortex...')
            self.PS_shear = vortex_detector(
                'PS_shear', PS_shear_windowLL, PS_shear_windowUR,
                self.data['y'], self.data['z'], mean_u, 
                mean_vort_x * self.chord / self.velocity, 
                choice='area', level=shear_level
            )
            self.loc_points_aux.append(self.PS_shear)
            aux_vortex_count += 1
            print(f'        Primary shear vortex core: {self.PS_shear.core.core_loc[0]}')
        
        print(f'    Vortex detection complete: 2 primary vortices + {aux_vortex_count} auxiliary vortices detected.')
    
    def extract_invariants(self):
        print('\n----> Extracting velocity invariants...')
        
        # Extract invariants for primary vortex
        print('    Extracting primary vortex invariants...')
        print('        Radius: 0.01, Layers: 2, Angle range: 0-180°')
        self.loc_points_PV = extract_velocity_invariants(
            self.data, self.connectivity, self.P_Vortex, 
            location=self.location, Vortex_Type="PV", dt = self.dt,
            radius=0.01, n_layers=2, start_angle=0, end_angle=180,
            data_type=self.data_type, velocity = self.velocity, angle_of_attack = self.angle_of_attack,
            limited_gradient=self.limited_gradient
        )
        
        # Extract invariants for secondary vortex
        print('    Extracting secondary vortex invariants...')
        print('        Radius: 0.007, Layers: 2, Angle range: -90-90°')
        self.loc_points_SV = extract_velocity_invariants(
            self.data, self.connectivity, self.S_Vortex, 
            location=self.location, Vortex_Type="SV", dt = self.dt,
            radius=0.002, n_layers=1, start_angle=-90, end_angle=90,
            data_type=self.data_type, velocity = self.velocity, angle_of_attack = self.angle_of_attack,
            limited_gradient=self.limited_gradient
        )
        
        # Extract invariants for auxiliary vortices
        self.loc_points_aux_coords = []
        aux_extracted = 0
        
        if hasattr(self, 'T_Vortex'):
            print('    Extracting tertiary vortex invariants...')
            print('        Radius: 0.007, Layers: 2, Angle range: 0-180°')
            loc_points_TV = extract_velocity_invariants(
                self.data, self.connectivity, self.T_Vortex, 
                location=self.location, Vortex_Type="TV", dt = self.dt,
                radius=0.007, n_layers=2, start_angle=0, end_angle=180,
                data_type=self.data_type, velocity = self.velocity, angle_of_attack = self.angle_of_attack,
                limited_gradient=self.limited_gradient
            )
            self.loc_points_aux_coords.append(loc_points_TV[:, 0])
            aux_extracted += 1
        
        if hasattr(self, 'SS_shear'):
            print('    Extracting secondary shear vortex invariants...')
            # Use larger radius for PIV shear vortices
            shear_radius = 0.003 if self.data_type == 'PIV' else 0.0005
            print(f'        Radius: {shear_radius}, Layers: 2, Angle range: -90-90°')
            loc_points_SS = extract_velocity_invariants(
                self.data, self.connectivity, self.SS_shear, 
                location=self.location, Vortex_Type="SS_shear", dt = self.dt,
                radius=shear_radius, n_layers=2, start_angle=-90, end_angle=90,
                data_type=self.data_type, velocity = self.velocity, angle_of_attack = self.angle_of_attack,
                limited_gradient=self.limited_gradient
            )
            self.loc_points_aux_coords.append(loc_points_SS[:, 0])
            aux_extracted += 1
        
        if hasattr(self, 'PS_shear'):
            print('    Extracting primary shear vortex invariants...')
            # Use larger radius for PIV shear vortices
            shear_radius = 0.003 if self.data_type == 'PIV' else 0.0005
            print(f'        Radius: {shear_radius}, Layers: 2, Angle range: 0-180°')
            loc_points_PS = extract_velocity_invariants(
                self.data, self.connectivity, self.PS_shear, 
                location=self.location, Vortex_Type="PS_shear", dt = self.dt,
                radius=shear_radius, n_layers=2, start_angle=0, end_angle=180,
                data_type=self.data_type, velocity = self.velocity, angle_of_attack = self.angle_of_attack,
                limited_gradient=self.limited_gradient
            )
            self.loc_points_aux_coords.append(loc_points_PS[:, 0])
            aux_extracted += 1
        
        print(f'    Invariant extraction complete: 2 primary + {aux_extracted} auxiliary vortices processed.')
    
    def generate_plots(self):
        print('\n----> Generating plots...')
        
        # Plot global invariants
        print('    Generating global invariant plots...')
        plot_global_invariants(
            self.data_grid, self.chord, self.location, 
            self.loc_points_PV, self.loc_points_SV, self.loc_points_aux_coords,
            self.data_type, limited_gradient = self.limited_gradient
        )
        print('        Global invariant plots saved.')
        
        # Clean up grid data
        del self.data_grid
        
        # Plot vortex profiles
        print('    Generating vortex profile plots...')
        grid = [self.data['y'], self.data['z']]
        
        print(f'        Primary vortex PCA profile (points: {self.pca_points}, length: {self.pca_length})...')
        plot_vortex_profiles(
            'primary', self.location, grid, self.connectivity, 
            self.P_Vortex, self.S_Vortex,
            Qhat_all=self.data['Qhat_all'], Rhat_all=self.data['Rhat_all'],
            Qs_all=self.data['Qs_all'], Rs_all=self.data['Rs_all'],
            Qw_all=self.data['Qw_all'], var_S=self.data['var_S'], 
            num_query_points=self.pca_points, L=self.pca_length, 
            data_type=self.data_type, limited_gradient = self.limited_gradient
        )
        
        print(f'        Secondary vortex PCA profile (points: {self.pca_points}, length: {self.pca_length})...')
        plot_vortex_profiles(
            'secondary', self.location, grid, self.connectivity, 
            self.P_Vortex, self.S_Vortex,
            Qhat_all=self.data['Qhat_all'], Rhat_all=self.data['Rhat_all'],
            Qs_all=self.data['Qs_all'], Rs_all=self.data['Rs_all'],
            Qw_all=self.data['Qw_all'], var_S=self.data['var_S'], 
            num_query_points=self.pca_points, L=self.pca_length, 
            data_type=self.data_type, limited_gradient = self.limited_gradient
        )
        
        if self.data_type == 'LES' and self.limited_gradient == False:
            print('    Generating Spectral Plots...')
            self.spectra_file = "Velocity_Spectra_Core_B_{}AOA_U{}_{}.h5".format(self.angle_of_attack, self.velocity, self.data_type)
            plot_spectra(self.spectra_file, self.location, self.data_type, nchunk=8, xlim_vel=(50,5e4), xlim_p=(50,5e4), xlim_vort=(50,5e4), xlim_tke=(50,5e4))
            print('    Plot generation complete.')
        
    def generate_combined_plots(self):
        """Generate combined Q-R plots across multiple locations."""
        self.spectra_file = "Velocity_Spectra_Core_B_{}AOA_U{}_{}.h5".format(self.angle_of_attack, self.velocity, self.data_type)
        print('\n----> Generating combined Q-R plots along core...')
        print(f'    Locations: {self.locations}')
        print(f'    Data type: {self.data_type}')
        
        # Create output directory name with data type and limited gradient suffix
        output_dir = f'QR_Plots_{self.data_type}'
        if self.limited_gradient and self.data_type == 'LES':
            output_dir += '_Limited'
        
        plot_combine_qr(locations=self.locations,data_type=self.data_type,velocity=self.velocity,angle_of_attack=self.angle_of_attack,
            bins=100,output_dir=output_dir,limited_gradient=self.limited_gradient)
        
        print('    Combined Q-R plot generation complete.')
        
        if self.limited_gradient == False and self.data_type == 'LES' and self.plot_spectra == True:
            print('\n----> Generating combined spectral plots along core...')
            print(f'    Locations: {self.locations}')
            print(f'    Spectra File: {self.spectra_file}')
            plot_combine_spectra(data_file = self.spectra_file,locations = self.locations,variables=('u','v','w','pressure','vort_x'),
                num_features=825, nchunk=3, output_dir = output_dir, plot_ref_slope=True)
            print('    Combined spectra plot generation complete.')
    
    @timer
    def run(self):
        print(f"\n{'='*100}")
        print(f"{'Performing vortex plot analysis.':^100}")
        print(f"{'='*100}\n")
        
        # Print input parameters
        print('\n----> Input Parameters:')
        print(f'    Cut location: {self.location}')
        print(f'    Data type: {self.data_type}')
        print(f'    Output directory: {self.output_dir}')
        print(f'    Chord length: {self.chord} m')
        print(f'    Free stream velocity: {self.velocity} m/s')
        print(f'    Angle of attack: {self.angle_of_attack}°')
        print(f'    Grid resolution: {self.grid_size} x {self.grid_size}')
        print(f'    PCA query points: {self.pca_points}')
        print(f'    PCA line length: {self.pca_length} m')
        print(f'    Plot all: {self.plot_all}')
        print(f'    Limited gradient (LES only): {self.limited_gradient}')
        print(f'    Time step between frames (LES only): {self.dt} s')
        print(f'    Plot spectra (LES only): {self.plot_spectra}')
        print(f'    Locations: {self.locations}')
        
        # Setup plotting parameters
        print('\n----> Setting up plotting parameters...')
        _setup_plot_params()
        print('    Matplotlib parameters configured.')
        
        # Main processing steps
        if self.plot_all:
            # Generate only combined plots across locations
            self.generate_combined_plots()
        else:
            # Generate only single-location plots
            self.load_data()
            self.create_grid()
            self.detect_vortices()
            self.extract_invariants()
            self.generate_plots()
        
        print(f"\n{'='*100}")
        print(f"{'Vortex plot analysis complete.':^100}")
        print(f"{'='*100}\n")