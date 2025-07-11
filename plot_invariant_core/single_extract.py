import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from itertools import combinations
from scipy.ndimage import binary_dilation
from sklearn.decomposition import PCA
from .utils import setup_rcparams, print, create_directory_if_not_exists
from .grid_maker import make_grid
from .vortex_detector import vortex, find_squares
# Import window boundaries from the main module
import sys
sys.path.append('..')
from window_bounds import get_window_boundaries

def extract_single_qr_data(data, connectivity, cut, data_type, vortex_detect_dir, 
                          velocity_invariant_dir, chord, velocity, angle_of_attack):
    """Extract and plot single location QR data."""
    setup_rcparams()
    
    print(f"    Processing single location QR extraction for {cut}")
    
    # Create grid for global invariants plotting
    y_bnd = [-0.05, 0.05]
    z_bnd = [-0.16, -0.06]
    
    # Construct the grid for plotting global invariants
    data_grid = make_grid(500, y_bnd, z_bnd, data['y'], data['z'], 
                         np.mean(data['u'], axis=1), np.mean(data['vort_x'], axis=1), 'vort_x')
    
    # Add other variables to grid
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['Phat_all'], axis=1), 'Phat')
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['Qhat_all'], axis=1), 'Qhat')
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['Rhat_all'], axis=1), 'Rhat')
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['pressure_hessian'], axis=1), 'Pressure_Hessian')
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['Qs_all'], axis=1), 'Qs')
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['Rs_all'], axis=1), 'Rs')
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['Qw_all'], axis=1), 'Qw')
    data_grid.calculate_grid(500, y_bnd, z_bnd, data['var_A'], 'var_A')
    data_grid.calculate_grid(500, y_bnd, z_bnd, data['var_S'], 'var_S')
    data_grid.calculate_grid(500, y_bnd, z_bnd, data['var_omega'], 'var_omega')
    data_grid.calculate_grid(500, y_bnd, z_bnd, data['mean_SR'], 'mean_SR')
    
    # Get window boundaries
    boundaries = get_window_boundaries(cut, str(int(angle_of_attack)))
    
    # Extract vortex locations
    mean_vort_x = np.mean(data['vort_x'], axis=1)
    
    # Create vortex objects
    vortices = {}
    loc_points = {}
    
    # Primary vortex
    if boundaries['PV_WindowLL'] and boundaries['PV_WindowUR']:
        vortices['PV'] = vortex('Primary', boundaries['PV_WindowLL'], boundaries['PV_WindowUR'], 
                               data['y'], data['z'], np.mean(data['u'], axis=1), 
                               mean_vort_x * chord / velocity, choice='precise', level=-35)
        loc_points['PV'] = extract_velocity_invariants(
            data, connectivity, vortices['PV'], cut, "PV", 
            radius=0.01, n_layers=2, start_angle=0, end_angle=180, data_type=data_type)
    
    # Secondary vortex
    if boundaries['SV_WindowLL'] and boundaries['SV_WindowUR']:
        vortices['SV'] = vortex('Secondary', boundaries['SV_WindowLL'], boundaries['SV_WindowUR'], 
                               data['y'], data['z'], np.mean(data['u'], axis=1), 
                               mean_vort_x * chord / velocity, choice='precise', level=-35)
        loc_points['SV'] = extract_velocity_invariants(
            data, connectivity, vortices['SV'], cut, "SV", 
            radius=0.007, n_layers=2, start_angle=-90, end_angle=90, data_type=data_type)
    
    # Additional vortices
    if boundaries.get('TV_WindowLL') and boundaries.get('TV_WindowUR'):
        vortices['TV'] = vortex('Tertiary', boundaries['TV_WindowLL'], boundaries['TV_WindowUR'], 
                               data['y'], data['z'], np.mean(data['u'], axis=1), 
                               mean_vort_x * chord / velocity, choice='area', level=-30)
        loc_points['TV'] = extract_velocity_invariants(
            data, connectivity, vortices['TV'], cut, "TV", 
            radius=0.007, n_layers=2, start_angle=0, end_angle=180, data_type=data_type)
    
    # Plot global invariants
    plot_global_invariants_single(data_grid, chord, cut, loc_points, data_type, velocity)
    
    # Plot vortex profiles if vortices were found
    if 'PV' in vortices:
        plot_vortex_profiles_single('primary', cut, [data['y'], data['z']], connectivity,
                                   data, vortices['PV'], data_type=data_type)
    
    if 'SV' in vortices:
        plot_vortex_profiles_single('secondary', cut, [data['y'], data['z']], connectivity,
                                   data, vortices['SV'], data_type=data_type)
    
    print(f"    Completed single location QR extraction for {cut}")


def extract_velocity_invariants(data, connectivity, vortex_obj, location, vortex_type, 
                               radius=0.01, n=6, n_layers=2, start_angle=0, end_angle=180, data_type='LES'):
    """Extract velocity invariants at vortex core and adjacent cells."""
    print(f"        Processing {vortex_type} vortex at location: {location}")
    
    grid = [data['y'], data['z']]
    loc_points, closest_indices, adjacent_points_list = find_closest_indices_and_adjacent_cells(
        vortex_obj.core.core_loc[0], grid, connectivity, 
        n=n, radius=radius, n_layers=n_layers, start_angle=start_angle, end_angle=end_angle)
    
    # Extract invariant data
    Phat = extract_variable_data_stacked(data['Phat_all'], closest_indices, adjacent_points_list)
    Qhat = extract_variable_data_stacked(data['Qhat_all'], closest_indices, adjacent_points_list)
    Rhat = extract_variable_data_stacked(data['Rhat_all'], closest_indices, adjacent_points_list)
    Rs = extract_variable_data_stacked(data['Rs_all'], closest_indices, adjacent_points_list)
    Qs = extract_variable_data_stacked(data['Qs_all'], closest_indices, adjacent_points_list)
    Qw = extract_variable_data_stacked(data['Qw_all'], closest_indices, adjacent_points_list)
    
    # Extract variance data
    var_A = extract_mean_variable(data['var_A'], closest_indices, adjacent_points_list)
    var_S = extract_mean_variable(data['var_S'], closest_indices, adjacent_points_list)
    var_omega = extract_mean_variable(data['var_omega'], closest_indices, adjacent_points_list)
    mean_SR = extract_mean_variable(data['mean_SR'], closest_indices, adjacent_points_list)
    
    # Normalize strain-rotation invariants
    Rs = Rs / (var_S[:, np.newaxis] ** (3/2))
    Qs = Qs / (var_S[:, np.newaxis])
    Qw = Qw / (var_S[:, np.newaxis])
    
    # Plot local invariants
    plot_local_invariants_QR(location, Rhat, Qhat, vortex_type, data_type)
    plot_local_invariants_Qs_Rs(location, Rs, Qs, vortex_type, data_type)
    plot_local_invariants_Qs_Qw(location, Qw, Qs, vortex_type, data_type)
    
    # Save extracted data
    save_extracted_data(location, Phat, Qhat, Rhat, Qs, Qw, Rs, vortex_type)
    
    return loc_points

def find_closest_indices_and_adjacent_cells(core_loc, grid, connectivity, n=6, radius=0.01,
                                           start_angle=0, end_angle=180, n_layers=1):
    """Find closest indices and adjacent cells around vortex core."""
    angles = np.linspace(start_angle * np.pi / 180, end_angle * np.pi / 180, n - 1, endpoint=True)
    x = core_loc[0] + radius * np.cos(angles)
    y = core_loc[1] + radius * np.sin(angles)
    x = np.insert(x, 0, core_loc[0])
    y = np.insert(y, 0, core_loc[1])
    loc_points = np.array([x, y])
    
    closest_indices = []
    adjacent_points_list = []
    
    for i in range(loc_points.shape[1]):
        dist = np.sqrt((grid[0] - loc_points[0, i])**2 + (grid[1] - loc_points[1, i])**2)
        closest_index = np.argmin(dist)
        closest_indices.append(closest_index)
        
        # Find adjacent cells
        rows = np.where(np.any(connectivity == closest_index, axis=1))[0]
        if len(rows) == 0:
            adjacent = set()
        else:
            chosen_row = connectivity[rows[0], :]
            adjacent = set(chosen_row)
            adjacent.discard(closest_index)
        
        # Add multiple layers if requested
        for layer in range(1, n_layers):
            new_adjacent = set()
            for pt in adjacent:
                rows_pt = np.where(np.any(connectivity == pt, axis=1))[0]
                if len(rows_pt) > 0:
                    chosen_row_pt = connectivity[rows_pt[0], :]
                    new_adjacent.update(chosen_row_pt)
            adjacent.update(new_adjacent)
            adjacent.discard(closest_index)
        
        adjacent_points_list.append(adjacent)
    
    return loc_points, closest_indices, adjacent_points_list

def extract_variable_data_stacked(variable_all, core_indices, adjacent_points_list):
    """Extract variable data stacked for analysis."""
    T = variable_all.shape[1]
    stacked_list = []
    
    for core, adj_set in zip(core_indices, adjacent_points_list):
        cell_indices = [core] + sorted(list(adj_set))
        data = variable_all[cell_indices, :]
        stacked = data.reshape(-1)
        stacked_list.append(stacked)
    
    return np.vstack(stacked_list)

def extract_mean_variable(variable_all, core_indices, adjacent_points_list):
    """Extract mean variable for normalization."""
    averaged_list = []
    
    for core, adj_set in zip(core_indices, adjacent_points_list):
        cell_indices = [core] + sorted(list(adj_set))
        data = variable_all[cell_indices]
        avg_value = np.mean(data)
        averaged_list.append(avg_value)
    
    return np.array(averaged_list)

def plot_local_invariants_QR(location, R_hat, Q_hat, vortex_type, data_type):
    """Plot local QR invariants."""
    setup_rcparams()
    bins = 100
    num_points = int(np.shape(R_hat)[0])
    
    fig, axs = plt.subplots(int(num_points/2), 2, figsize=(int(num_points/2)*3+1, 8))
    axs = axs.flatten()
    
    for i in range(num_points):
        R_vals = R_hat[i, :]
        Q_vals = Q_hat[i, :]
        
        # Compute histogram
        h, edgesR, edgesQ = np.histogram2d(R_vals, Q_vals, bins=bins)
        totalPoints = R_vals.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Center edges
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)
        
        # Set limits
        x_lim = max(np.abs(np.mean(R_vals) + 2 * np.std(R_vals)),
                    np.abs(np.mean(R_vals) - 2 * np.std(R_vals)))
        y_lim = max(np.abs(np.mean(Q_vals) + 2 * np.std(Q_vals)),
                    np.abs(np.mean(Q_vals) - 2 * np.std(Q_vals)))
        axs[i].set_xlim(-x_lim, x_lim)
        axs[i].set_ylim(-y_lim, y_lim)
        
        # Normalize and filter
        pdf_norm = pdf / np.max(pdf)
        pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
        
        # Plot contour
        color_map = plt.cm.hot.reversed()
        c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                           cmap=color_map, extend='both')
        axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                      colors='blue', linestyles='dashed', linewidths=1)
        
        # Add reference lines
        x_zero_pdf = np.linspace(-x_lim, x_lim, 1000)
        Q_zero_pdf = -3 * ((x_zero_pdf / 2)**2)**(1/3)
        axs[i].axvline(0, color='black', linestyle='--', linewidth=2)
        axs[i].plot(x_zero_pdf, Q_zero_pdf, '--k', linewidth=2)
        
        # Labels
        axs[i].set_xlabel(r'$\hat{R}$', fontsize=14)
        axs[i].set_ylabel(r'$\hat{Q}$', fontsize=14)
        axs[i].set_title(f'Point {i+1}', fontsize=16)
        axs[i].tick_params(labelsize=12)
    
    fig.tight_layout()
    plt.savefig(f'Velocity_Invariants_{location}_{data_type}/Local_Velocity_Invariants_QR_{location}_{vortex_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_local_invariants_Qs_Rs(location, Rs_hat, Qs_hat, vortex_type, data_type):
    """Plot local Qs-Rs invariants."""
    setup_rcparams()
    bins = 100
    a_values = [-0.5, 0, 1/2, 1/3, 1]
    num_points = int(np.shape(Rs_hat)[0])
    
    fig, axs = plt.subplots(int(num_points/2), 2, figsize=(int(num_points/2)*3+1, 8))
    axs = axs.flatten()
    
    for i in range(num_points):
        R_vals = Rs_hat[i, :]
        Q_vals = Qs_hat[i, :]
        
        # Compute histogram
        h, edgesR, edgesQ = np.histogram2d(R_vals, Q_vals, bins=bins)
        totalPoints = R_vals.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Center edges
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)
        
        # Set limits
        x_lim = max(np.abs(np.mean(R_vals) + 2 * np.std(R_vals)),
                    np.abs(np.mean(R_vals) - 2 * np.std(R_vals)))
        y_lim = max(np.abs(np.mean(Q_vals) + 2 * np.std(Q_vals)),
                    np.abs(np.mean(Q_vals) - 2 * np.std(Q_vals)))
        axs[i].set_xlim(-x_lim, x_lim)
        axs[i].set_ylim(-y_lim, 0)
        
        # Normalize and filter
        pdf_norm = pdf / np.max(pdf)
        pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
        
        # Plot theoretical lines
        Q_plus = np.linspace(0.001, y_lim, 1000)
        for a in a_values:
            RS_line = - (Q_plus**(3/2)) * a * (1+a) / ((1+a+a**2)**(3/2))
            axs[i].plot(-RS_line, -Q_plus, '--', linewidth=2)
        
        # Plot contour
        color_map = plt.cm.hot.reversed()
        c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                           cmap=color_map, extend='both')
        axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                      colors='blue', linestyles='dashed', linewidths=1)
        
        # Labels
        axs[i].grid(True, which='both', linestyle='--', alpha=0.5)
        axs[i].set_xlabel(r'$\hat{R}_s$', fontsize=14)
        axs[i].set_ylabel(r'$\hat{Q}_s$', fontsize=14)
        axs[i].set_title(f'Point {i+1}', fontsize=16)
        axs[i].tick_params(labelsize=12)
    
    fig.tight_layout()
    plt.savefig(f'Velocity_Invariants_{location}_{data_type}/Local_Velocity_Invariants_Qs_Rs_{location}_{vortex_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_local_invariants_Qs_Qw(location, Qw_hat, Qs_hat, vortex_type, data_type):
    """Plot local Qs-Qw invariants."""
    setup_rcparams()
    bins = 100
    num_points = int(np.shape(Qs_hat)[0])
    
    fig, axs = plt.subplots(int(num_points/2), 2, figsize=(int(num_points/2)*3+1, 8))
    axs = axs.flatten()
    
    for i in range(num_points):
        R_vals = Qw_hat[i, :]
        Q_vals = -Qs_hat[i, :]
        
        # Compute histogram
        h, edgesR, edgesQ = np.histogram2d(R_vals, Q_vals, bins=bins)
        totalPoints = R_vals.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Center edges
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)
        
        # Set limits
        x_lim = max(np.abs(np.mean(R_vals) + 2 * np.std(R_vals)),
                    np.abs(np.mean(R_vals) - 2 * np.std(R_vals)))
        y_lim = max(np.abs(np.mean(Q_vals) + 2 * np.std(Q_vals)),
                    np.abs(np.mean(Q_vals) - 2 * np.std(Q_vals)))
        axs[i].set_xlim(0, x_lim)
        axs[i].set_ylim(0, y_lim)
        
        # Normalize and filter
        pdf_norm = pdf / np.max(pdf)
        pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
        
        # Plot contour
        x_zero_pdf = np.linspace(x_pdf[0, 0], x_pdf[0, -1], 1000)
        Q_zero_pdf = x_zero_pdf
        
        cs = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                            cmap=plt.cm.hot.reversed(), extend='both')
        axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2],
                      colors='blue', linestyles='dashed', linewidths=1)
        
        # Add reference lines
        axs[i].axvline(0, color='black', linestyle='--', linewidth=2)
        axs[i].plot(x_zero_pdf, Q_zero_pdf, '--k', linewidth=2)
        
        # Labels
        axs[i].set_xlabel(r'$\hat{Q}_w$', fontsize=14)
        axs[i].set_ylabel(r'$-\hat{Q}_s$', fontsize=14)
        axs[i].set_title(f'Point {i+1}', fontsize=16)
        axs[i].tick_params(labelsize=12)
    
    fig.tight_layout()
    plt.savefig(f'Velocity_Invariants_{location}_{data_type}/Local_Velocity_Invariants_Qs_Qw_{location}_{vortex_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_global_invariants_single(data_grid, chord, location, loc_points, data_type, velocity):
    """Plot global invariants for single location."""
    # Implementation similar to the original but simplified
    print(f"        Plotting global invariants for {location}")
    # This would contain the global plotting logic from the original Single_QR_Extract.py
    pass

def plot_vortex_profiles_single(vortex_type, location, grid, connectivity, data, vortex_obj, data_type='LES'):
    """Plot vortex profiles along PCA axis for single location."""
    # Implementation similar to the original but simplified
    print(f"        Plotting {vortex_type} vortex profiles for {location}")
    # This would contain the PCA profile plotting logic from the original Single_QR_Extract.py
    pass

def save_extracted_data(location, P_hat, Q_hat, R_hat, Qs_hat, Qw_hat, Rs_hat, vortex_type):
    """Save extracted data to combined HDF5 file."""
    filename = 'Velocity_Invariants_Core_B_10AOA_LES_U30.h5'
    
    with h5py.File(filename, 'a') as f:
        # Ensure the location group exists
        if location not in f:
            f.create_group(location)
        
        # Create the vortex_type subgroup under the location group
        if vortex_type not in f[location]:
            f[location].create_group(vortex_type)
        
        # Save each dataset under the vortex_type subgroup
        for name, dat in zip(['P', 'Q', 'R', 'Qs', 'Qw', 'Rs'],
                            [P_hat, Q_hat, R_hat, Qs_hat, Qw_hat, Rs_hat]):
            # If the dataset already exists, remove it
            if name in f[location][vortex_type]:
                del f[location][vortex_type][name]
            f[location][vortex_type].create_dataset(name, data=dat)
    
    print(f"        Saved extracted data for {location} {vortex_type}")