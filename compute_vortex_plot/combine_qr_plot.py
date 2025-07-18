# Combined plot for the Vortex invariants QR PDF contours at the vortex core locations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.ndimage import gaussian_filter
import os
from itertools import combinations
from scipy.ndimage import binary_dilation
import h5py
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from .utils import print


def setup_plot_params():
    """Setup matplotlib parameters for consistent plot styling."""
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


def get_data_file_path(data_type, velocity=30, angle_of_attack=10, limited_gradient=False):
    """Get the appropriate data file path based on data type and limited_gradient option."""
    base_path = os.getcwd()
    if data_type.upper() == 'LES':
        data_path = f'Velocity_Invariants_Core_B_{angle_of_attack}AOA_LES_U{velocity}.h5'
        if limited_gradient:
            data_path = data_path.replace('.h5', '_Limited.h5')
    elif data_type.upper() == 'PIV':
        data_path = f'Velocity_Invariants_Core_B_{angle_of_attack}AOA_PIV_U{velocity}.h5'
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'LES' or 'PIV'.")
    
    data_file = os.path.join(base_path, data_path)
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    return data_file


def extract_QR_data(data_file, locations, vortex, num_features):
    """Extract Q, R, Qs, Rs, Qw data from HDF5 file for specified locations and vortex type."""
    q, qs, qw, r, rs = [], [], [], [], []
    
    with h5py.File(data_file, 'r') as file:
        for loc in locations:
            if loc in file:
                loc_group = file[loc]
                if vortex in loc_group:
                    subgroup = loc_group[vortex]
                    if num_features is None:
                        num_features = subgroup['Q'].shape[1] if 'Q' in subgroup else 0
                    
                    for var_name, var_list in [('Q', q), ('Qs', qs), ('Qw', qw), ('R', r), ('Rs', rs)]:
                        if var_name in subgroup:
                            var_row = subgroup[var_name][0]
                            var_row = var_row[:num_features] if var_row.shape[0] >= num_features else np.pad(var_row, (0, num_features - var_row.shape[0]))
                            var_list.append(var_row)
                        else:
                            # If data doesn't exist, add zeros
                            var_list.append(np.zeros(num_features))
    
    return np.array(q), np.array(qs), np.array(qw), np.array(r), np.array(rs)


def plot_QR_along_vortex(q, r, bins=100, set_lim=None, vortex_group='PV', output_dir='QR_Plots'):
    """Plot Q-R invariant contours along vortex locations."""
    if r.shape[0] == 0:
        print(f"Warning: No data found for vortex group {vortex_group}")
        return
    
    fig, axs = plt.subplots(1, int(r.shape[0]), figsize=(int(r.shape[0]*2), 3), 
                           sharey=True, gridspec_kw={'wspace': 0})
    
    # Handle single subplot case
    if r.shape[0] == 1:
        axs = [axs]
    
    for i in range(r.shape[0]):
        Rhat = r[i, :]
        Qhat = q[i, :]
        
        # Skip if no valid data
        if len(Rhat) == 0 or np.all(Rhat == 0):
            continue
        
        # Compute 2D histogram
        h, edgesR, edgesQ = np.histogram2d(Rhat, Qhat, bins=bins)
        
        # Normalize the histogram
        totalPoints = Rhat.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Center the bin edges
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)

        # Set axis limits
        if set_lim is not None:
            axs[i].set_xlim(set_lim[0], set_lim[1])
            axs[i].set_ylim(set_lim[2], set_lim[3])
        else:
            axs[i].set_xlim(-0.25, 0.25)
            axs[i].set_ylim(-1, 1)
        
        # Normalize and apply Gaussian filter
        if np.max(pdf) > 0:
            pdf_norm = pdf / np.max(pdf)
            pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])

            # Plot contours
            color_map = plt.cm.hot.reversed()
            c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                               cmap=color_map, extend='both')
            axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                          colors='blue', linestyles='dashed', linewidths=1)

        # Delta=0 curve
        x_zero_pdf = np.linspace(-2, 2, 1000)
        Q_zero_pdf = -3 * ((x_zero_pdf / 2) ** 2) ** (1 / 3)

        # Plot reference lines
        axs[i].axvline(0, color='black', linestyle='--', linewidth=2)
        axs[i].plot(x_zero_pdf, Q_zero_pdf, '--k', linewidth=2)

        # Set labels
        if i < r.shape[0] - 1:
            axs[i].set_xticklabels([])
        axs[i].set_xlabel('$\hat{R}$')

    axs[0].set_ylabel('$\hat{Q}$')
    fig.tight_layout()
    
    # Save plot
    figname = os.path.join(output_dir, f'B_10AOA_{vortex_group}_Cores_horizontal')
    plt.savefig(figname + '.jpeg', format='jpeg', dpi=600)
    plt.savefig(figname + '.eps', format='eps', dpi=600)
    plt.close()


def plot_QsRs_along_vortex(qs, rs, bins=100, set_lim=None, vortex_group='PV', output_dir='QR_Plots'):
    """Plot Qs-Rs strain tensor invariant contours along vortex locations."""
    if rs.shape[0] == 0:
        print(f"Warning: No data found for vortex group {vortex_group}")
        return
    
    fig, axs = plt.subplots(1, int(rs.shape[0]), figsize=(int(rs.shape[0]*2), 3), 
                           sharey=True, gridspec_kw={'wspace': 0})
    
    # Handle single subplot case
    if rs.shape[0] == 1:
        axs = [axs]
    
    a_values = [-0.5, 0, 1/2, 1/3, 1]
    
    for i in range(rs.shape[0]):
        Rhat = rs[i, :]
        Qhat = qs[i, :]
        
        # Skip if no valid data
        if len(Rhat) == 0 or np.all(Rhat == 0):
            continue
        
        # Compute 2D histogram
        h, edgesR, edgesQ = np.histogram2d(Rhat, Qhat, bins=bins)
        
        # Normalize the histogram
        totalPoints = Rhat.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Center the bin edges
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)

        # Set axis limits
        if set_lim is not None:
            axs[i].set_xlim(set_lim[0], set_lim[1])
            axs[i].set_ylim(set_lim[2], set_lim[3])
        else:
            axs[i].set_xlim(-0.5, 0.5)
            axs[i].set_ylim(-1.5, 0)
            
        # Normalize and apply Gaussian filter
        if np.max(pdf) > 0:
            pdf_norm = pdf / np.max(pdf)
            pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
            
            # Plot contours
            color_map = plt.cm.hot.reversed()
            c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                               cmap=color_map, extend='both')
            axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                          colors='blue', linestyles='dashed', linewidths=1)

        # Plot strain tensor lines
        y_max = axs[i].get_ylim()[1]
        Q_plus = np.linspace(0.001, -y_max, 1000)
        
        for a in a_values:
            RS_line = -(Q_plus**(3/2)) * a * (1+a) / ((1+a+a**2)**(3/2))
            axs[i].plot(-RS_line, -Q_plus, '--', linewidth=2)

        # Set labels
        if i < rs.shape[0] - 1:
            axs[i].set_xticklabels([])
        axs[i].set_xlabel('$\hat{R}_s$')

    axs[0].set_ylabel('$\hat{Q}_s$')
    fig.tight_layout()
    
    # Save plot
    figname = os.path.join(output_dir, f'B_10AOA_{vortex_group}_Cores_Qs_Rs_horizontal')
    plt.savefig(figname + '.eps', format='eps', dpi=600)
    plt.savefig(figname + '.jpeg', format='jpeg', dpi=600)
    plt.close()


def plot_QsQw_along_vortex(qs, qw, bins=100, set_lim=None, vortex_group='PV', output_dir='QR_Plots'):
    """Plot Qs-Qw strain-vorticity invariant contours along vortex locations."""
    if qs.shape[0] == 0:
        print(f"Warning: No data found for vortex group {vortex_group}")
        return
    
    fig, axs = plt.subplots(1, int(qs.shape[0]), figsize=(int(qs.shape[0]*2), 3), 
                           sharey=True, gridspec_kw={'wspace': 0})
    
    # Handle single subplot case
    if qs.shape[0] == 1:
        axs = [axs]
    
    for i in range(qs.shape[0]):
        Rhat = qw[i, :]
        Qhat = -qs[i, :]
        
        # Skip if no valid data
        if len(Rhat) == 0 or np.all(Rhat == 0):
            continue
        
        # Define ranges
        Rmin, Rmax = 0, np.max(Rhat) if np.max(Rhat) > 0 else 1
        Qmin, Qmax = np.min(Qhat), np.max(Qhat)
        
        # Compute 2D histogram
        h, edgesR, edgesQ = np.histogram2d(Rhat, Qhat, bins=bins, 
                                          range=[[Rmin, Rmax], [Qmin, Qmax]])
        
        # Normalize the histogram
        totalPoints = Rhat.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Center the bin edges
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)

        # Set axis limits
        if set_lim is not None:
            axs[i].set_xlim(set_lim[0], set_lim[1])
            axs[i].set_ylim(set_lim[2], set_lim[3])
        else:
            axs[i].set_xlim(0, 1.5)
            axs[i].set_ylim(0, 1.5)

        # Normalize and apply Gaussian filter
        if np.max(pdf) > 0:
            pdf_norm = pdf / np.max(pdf)
            pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
            
            # Plot contours
            color_map = plt.cm.hot.reversed()
            c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                               cmap=color_map, extend='both')
            axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                          colors='blue', linestyles='dashed', linewidths=1)

        # Plot reference line
        x_zero_pdf = np.linspace(x_pdf[0, 0], x_pdf[0, -1], 1000)
        Q_zero_pdf = x_zero_pdf
        axs[i].plot(x_zero_pdf, Q_zero_pdf, '--k', linewidth=2, zorder=10)

        # Set labels
        if i < qs.shape[0] - 1:
            axs[i].set_xticklabels([])
        axs[i].set_xlabel('$\hat{Q}_w$')

    axs[0].set_ylabel('$-\hat{Q}_s$')
    fig.tight_layout()

    # Save plot
    figname = os.path.join(output_dir, f'B_10AOA_{vortex_group}_Cores_Qw_Qs_horizontal')
    plt.savefig(figname + '.eps', format='eps', dpi=600)
    plt.savefig(figname + '.jpeg', format='jpeg', dpi=600)
    plt.close()


def generate_combined_qr_plots(locations, data_type='LES', velocity=30, angle_of_attack=10, 
                             bins=100, output_dir='QR_Plots', limited_gradient=False):
    """
    Generate combined Q-R plots for all vortex types across specified locations.
    
    Parameters:
    -----------
    locations : list
        List of cutplane locations (e.g., ['030_TE', 'PIV1', 'PIV2', '085_TE', '095_TE', 'PIV3'])
    data_type : str
        Data type ('LES' or 'PIV')
    velocity : int
        Free stream velocity
    angle_of_attack : int
        Angle of attack in degrees
    bins : int
        Number of histogram bins
    output_dir : str
        Output directory for plots
    limited_gradient : bool
        Use limited gradient computation for LES data
    """
    print(f"\n----> Generating combined Q-R plots for {data_type} data...")
    print(f"    Locations: {locations}")
    print(f"    Output directory: {output_dir}")
    
    # Setup plotting parameters
    setup_plot_params()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data file path
    try:
        data_file = get_data_file_path(data_type, velocity, angle_of_attack, limited_gradient)
        print(f"    Using data file: {os.path.basename(data_file)}")
        if limited_gradient and data_type.upper() == 'LES':
            print(f"    Limited gradient mode enabled for LES data")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        return
    
    # Extract the max number of datapoints without truncation
    num_features = None
    
    # Define vortex types and their plot limits
    vortex_configs = {
        'PV': {'set_lim': [-0.25, 0.25, -1, 1]},
        'SV': {'set_lim': [-1.5, 1.5, -1, 8]},
        'TV': {'set_lim': [-0.25, 0.25, -1, 1]},
        'SS_shear': {'set_lim': [-0.25, 0.25, -1, 1]},
        'PS_shear': {'set_lim': [-0.25, 0.25, -1, 1]}
    }
    
    # Qs-Qw plot limits for different vortex types
    qsqw_limits = {
        'PV': [0, 2.5, 0, 1.5],
        'SV': [0, 10, 0, 2.5],
        'TV': [0, 2.5, 0, 1.5],
        'SS_shear': None,
        'PS_shear': None
    }
    
    plots_generated = 0
    
    for vortex_type, config in vortex_configs.items():
        print(f"    Processing {vortex_type} vortex...")
        
        # Extract data for this vortex type
        q, qs, qw, r, rs = extract_QR_data(data_file, locations, vortex_type, num_features)
        
        if q.size == 0:
            print(f"        No data found for {vortex_type}, skipping...")
            continue
        
        # Generate Q-R plots
        plot_QR_along_vortex(q, r, bins=bins, set_lim=config['set_lim'], 
                            vortex_group=vortex_type, output_dir=output_dir)
        plots_generated += 1
        
        # Generate Qs-Rs plots
        plot_QsRs_along_vortex(qs, rs, bins=bins, set_lim=None, 
                              vortex_group=vortex_type, output_dir=output_dir)
        plots_generated += 1
        
        # Generate Qs-Qw plots
        qsqw_lim = qsqw_limits.get(vortex_type)
        plot_QsQw_along_vortex(qs, qw, bins=bins, set_lim=qsqw_lim, 
                              vortex_group=vortex_type, output_dir=output_dir)
        plots_generated += 1
        
        print(f"        Generated 3 plot types for {vortex_type}")
    
    print(f"    Combined Q-R plot generation complete: {plots_generated} plots saved to {output_dir}/")