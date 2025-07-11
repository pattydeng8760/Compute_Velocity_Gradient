import os
import sys
import numpy as np
import argparse
from .utils import timer, print, init_logging_from_cut
from .data_loader import load_velocity_invariants, load_connectivity
from .grid_maker import make_grid
from .vortex_detector import vortex
from .invariant_extractor import extract_velocity_invariants
from .plotter import plot_global_invariants, plot_vortex_profiles
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
        
        # Create output directory
        self.output_dir = f'Velocity_Invariants_{self.location}_{self.data_type}'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_rcparams(self):
        import matplotlib.pyplot as plt
        SMALL_SIZE = 12
        MEDIUM_SIZE = 18
        LARGE_SIZE = 22
        plt.rcParams.update({
            'font.size': MEDIUM_SIZE,
            'axes.titlesize': MEDIUM_SIZE,
            'axes.labelsize': MEDIUM_SIZE,
            'xtick.labelsize': MEDIUM_SIZE,
            'ytick.labelsize': MEDIUM_SIZE,
            'legend.fontsize': SMALL_SIZE,
            'figure.titlesize': LARGE_SIZE,
            'mathtext.fontset': 'stix',
            'font.family': 'STIXGeneral',
        })
    
    def load_data(self):
        print('\n----> Loading velocity invariants data...')
        self.data = load_velocity_invariants(self.location, self.data_type)
        print('----> Loading connectivity data...')
        self.connectivity = load_connectivity(self.location, self.data_type)
        print('    Data loading complete.')
    
    def create_grid(self):
        print('\n----> Creating interpolation grid...')
        y_bnd = [-0.05, 0.05]
        z_bnd = [-0.16, -0.06]
        
        print(f'    Grid resolution: {self.grid_size} x {self.grid_size}')
        print(f'    Y bounds: {y_bnd}')
        print(f'    Z bounds: {z_bnd}')
        
        # Create grid for plotting global invariants
        self.data_grid = make_grid(
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
            ('Qs', np.mean(self.data['Qs_all'], axis=1)),
            ('Rs', np.mean(self.data['Rs_all'], axis=1)),
            ('Qw', np.mean(self.data['Qw_all'], axis=1)),
            ('var_A', self.data['var_A']),
            ('var_S', self.data['var_S']),
            ('var_omega', self.data['var_omega']),
            ('mean_SR', self.data['mean_SR'])
        ]
        
        for var_name, var_data in variables:
            self.data_grid.calculate_grid(self.grid_size, y_bnd, z_bnd, var_data, var_name)
        
        print('    Grid creation complete.')
    
    def detect_vortices(self):
        print('\n----> Detecting vortices...')
        
        # Get window boundaries
        boundaries = get_window_boundaries(self.location, str(self.angle_of_attack))
        print(f'    Window boundaries loaded for location: {self.location}')
        
        mean_vort_x = np.mean(self.data['vort_x'], axis=1)
        mean_u = np.mean(self.data['u'], axis=1)
        
        # Primary vortex
        print('    Detecting primary vortex...')
        self.P_Vortex = vortex(
            'Primary', boundaries['PV_WindowLL'], boundaries['PV_WindowUR'],
            self.data['y'], self.data['z'], mean_u, 
            mean_vort_x * self.chord / self.velocity, 
            choice='precise', level=-35
        )
        print(f'        Primary vortex core: {self.P_Vortex.core.core_loc[0]}')
        
        # Secondary vortex
        print('    Detecting secondary vortex...')
        self.S_Vortex = vortex(
            'Secondary', boundaries['SV_WindowLL'], boundaries['SV_WindowUR'],
            self.data['y'], self.data['z'], mean_u, 
            mean_vort_x * self.chord / self.velocity, 
            choice='precise', level=-35
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
            self.T_Vortex = vortex(
                'Tertiary', TV_WindowLL, TV_WindowUR,
                self.data['y'], self.data['z'], mean_u, 
                mean_vort_x * self.chord / self.velocity, 
                choice='area', level=-30
            )
            self.loc_points_aux.append(self.T_Vortex)
            aux_vortex_count += 1
            print(f'        Tertiary vortex core: {self.T_Vortex.core.core_loc[0]}')
        
        # Shear vortices
        SS_shear_windowLL = boundaries.get('SS_shear_windowLL')
        SS_shear_windowUR = boundaries.get('SS_shear_windowUR')
        if SS_shear_windowLL is not None and SS_shear_windowUR is not None and len(SS_shear_windowLL) == 2 and len(SS_shear_windowUR) == 2:
            print('    Detecting secondary shear vortex...')
            self.SS_shear = vortex(
                'SS_shear', SS_shear_windowLL, SS_shear_windowUR,
                self.data['y'], self.data['z'], mean_u, 
                mean_vort_x * self.chord / self.velocity, 
                choice='area', level=-15
            )
            self.loc_points_aux.append(self.SS_shear)
            aux_vortex_count += 1
            print(f'        Secondary shear vortex core: {self.SS_shear.core.core_loc[0]}')
        
        PS_shear_windowLL = boundaries.get('PS_shear_windowLL')
        PS_shear_windowUR = boundaries.get('PS_shear_windowLL')  # Note: original code has typo
        if PS_shear_windowLL is not None and PS_shear_windowUR is not None and len(PS_shear_windowLL) == 2 and len(PS_shear_windowUR) == 2:
            print('    Detecting primary shear vortex...')
            self.PS_shear = vortex(
                'PS_shear', PS_shear_windowLL, PS_shear_windowUR,
                self.data['y'], self.data['z'], mean_u, 
                mean_vort_x * self.chord / self.velocity, 
                choice='area', level=-15
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
            location=self.location, Vortex_Type="PV",
            radius=0.01, n_layers=2, start_angle=0, end_angle=180,
            data_type=self.data_type
        )
        
        # Extract invariants for secondary vortex
        print('    Extracting secondary vortex invariants...')
        print('        Radius: 0.007, Layers: 2, Angle range: -90-90°')
        self.loc_points_SV = extract_velocity_invariants(
            self.data, self.connectivity, self.S_Vortex, 
            location=self.location, Vortex_Type="SV",
            radius=0.007, n_layers=2, start_angle=-90, end_angle=90,
            data_type=self.data_type
        )
        
        # Extract invariants for auxiliary vortices
        self.loc_points_aux_coords = []
        aux_extracted = 0
        
        if hasattr(self, 'T_Vortex'):
            print('    Extracting tertiary vortex invariants...')
            print('        Radius: 0.007, Layers: 2, Angle range: 0-180°')
            loc_points_TV = extract_velocity_invariants(
                self.data, self.connectivity, self.T_Vortex, 
                location=self.location, Vortex_Type="TV",
                radius=0.007, n_layers=2, start_angle=0, end_angle=180,
                data_type=self.data_type
            )
            self.loc_points_aux_coords.append(loc_points_TV[:, 0])
            aux_extracted += 1
        
        if hasattr(self, 'SS_shear'):
            print('    Extracting secondary shear vortex invariants...')
            print('        Radius: 0.0005, Layers: 2, Angle range: -90-90°')
            loc_points_SS = extract_velocity_invariants(
                self.data, self.connectivity, self.SS_shear, 
                location=self.location, Vortex_Type="SS_shear",
                radius=0.0005, n_layers=2, start_angle=-90, end_angle=90,
                data_type=self.data_type
            )
            self.loc_points_aux_coords.append(loc_points_SS[:, 0])
            aux_extracted += 1
        
        if hasattr(self, 'PS_shear'):
            print('    Extracting primary shear vortex invariants...')
            print('        Radius: 0.0005, Layers: 2, Angle range: 0-180°')
            loc_points_PS = extract_velocity_invariants(
                self.data, self.connectivity, self.PS_shear, 
                location=self.location, Vortex_Type="PS_shear",
                radius=0.0005, n_layers=2, start_angle=0, end_angle=180,
                data_type=self.data_type
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
            self.data_type
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
            data_type=self.data_type
        )
        
        print(f'        Secondary vortex PCA profile (points: {self.pca_points}, length: {self.pca_length})...')
        plot_vortex_profiles(
            'secondary', self.location, grid, self.connectivity, 
            self.P_Vortex, self.S_Vortex,
            Qhat_all=self.data['Qhat_all'], Rhat_all=self.data['Rhat_all'],
            Qs_all=self.data['Qs_all'], Rs_all=self.data['Rs_all'],
            Qw_all=self.data['Qw_all'], var_S=self.data['var_S'], 
            num_query_points=self.pca_points, L=self.pca_length, 
            data_type=self.data_type
        )
        
        print('    Plot generation complete.')
    
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
        
        # Setup plotting parameters
        print('\n----> Setting up plotting parameters...')
        self.setup_rcparams()
        print('    Matplotlib parameters configured.')
        
        # Main processing steps
        self.load_data()
        self.create_grid()
        self.detect_vortices()
        self.extract_invariants()
        self.generate_plots()
        
        print(f"\n{'='*100}")
        print(f"{'Vortex plot analysis complete.':^100}")
        print(f"{'='*100}\n")