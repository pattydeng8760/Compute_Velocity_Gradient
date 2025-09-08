import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from matplotlib.ticker import EngFormatter, LogLocator, NullFormatter
from .utils import print, _next_greater_power_of_2, _welch_psd


def plot_global_invariants(data_grid, chord, location, loc_points_PV, loc_points_SV, loc_points_aux, data_type, limited_gradient=False):
    """
    Plot global velocity invariants on the interpolated grid.
    
    Parameters:
    -----------
    data_grid : object
        Grid object with interpolated data
    chord : float
        Chord length for normalization
    location : str
        Location identifier
    loc_points_PV : array
        Primary vortex location points
    loc_points_SV : array
        Secondary vortex location points
    loc_points_aux : list
        Auxiliary vortex location points
    data_type : str
        Data type ('LES' or 'PIV')
    """
    loc_points_aux = np.array(loc_points_aux)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    # --- Plot Phat ---
    cmap = plt.cm.bwr
    new_cmap = cmap(np.linspace(0, 1, 256))
    vmin, vmax = -1, 1
    lower_bound, upper_bound = -0.05, 0.05
    n_colors = len(new_cmap)
    lower_idx = int((lower_bound - vmin) / (vmax - vmin) * n_colors)
    upper_idx = int((upper_bound - vmin) / (vmax - vmin) * n_colors)
    new_cmap[lower_idx:upper_idx] = [1, 1, 1, 1]
    custom_cmap = mcolors.ListedColormap(new_cmap)
    
    cs1 = axs[0, 0].contourf(data_grid.grid_y / chord, 
                              data_grid.grid_z / chord + 0.1034 / chord, 
                              data_grid.Phat, levels=np.linspace(-1, 1, 201), 
                              cmap=custom_cmap, extend='both')
    axs[0, 0].set_title('$\hat{P}$')
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_xlabel('$y/c$')
    axs[0, 0].set_ylabel('$z/c$')
    
    # --- Plot Qhat ---
    bwr_cmap = plt.cm.bwr(np.linspace(0, 1, 256))
    new_colors = np.vstack(([1, 1, 1, 1], bwr_cmap, [1, 0, 0, 1]))
    custom_cmap = mcolors.ListedColormap(new_colors)
    
    cs2 = axs[0, 1].contourf(data_grid.grid_y / chord, 
                              data_grid.grid_z / chord + 0.1034 / chord, 
                              data_grid.Qhat, levels=np.linspace(-1, 1, 201), 
                              cmap=custom_cmap, extend='both')
    axs[0, 1].set_title('$\hat{Q}$')
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_xlabel('$y/c$')
    axs[0, 1].set_ylabel('$z/c$')
    
    # --- Plot Rhat ---
    cs3 = axs[0, 2].contourf(data_grid.grid_y / chord, 
                              data_grid.grid_z / chord + 0.1034 / chord, 
                              data_grid.Rhat, levels=np.linspace(-0.25, 0.25, 51), 
                              cmap=custom_cmap, extend='both')
    axs[0, 2].set_title('$\hat{R}$')
    axs[0, 2].set_aspect('equal')
    axs[0, 2].set_xlabel('$y/c$')
    axs[0, 2].set_ylabel('$z/c$')
    
    # --- Plot Vorticity ---
    cmap = plt.cm.bwr
    new_cmap = cmap(np.linspace(0, 1, 256))
    vmin, vmax = -100, 100
    lower_bound, upper_bound = -20, 20
    n_colors = len(new_cmap)
    lower_idx = int((lower_bound - vmin) / (vmax - vmin) * n_colors)
    upper_idx = int((upper_bound - vmin) / (vmax - vmin) * n_colors)
    new_cmap[lower_idx:upper_idx] = [1, 1, 1, 1]
    custom_cmap = mcolors.ListedColormap(new_cmap)
    cs4 = axs[1, 0].contourf(data_grid.grid_y / chord, 
                              data_grid.grid_z / chord + 0.1034 / chord, 
                              data_grid.vort_x*chord/30, levels=np.linspace(-120, 120, 201), 
                              cmap=custom_cmap, extend='both')
    axs[1, 0].set_title('$\Omega_x c/U_{\infty}$')
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_xlabel('$y/c$')
    axs[1, 0].set_ylabel('$z/c$')
    
    # Add scatter points for vortex locations
    axs[1, 0].scatter(loc_points_PV[0, :] / chord, loc_points_PV[1, :] / chord + 0.1034 / chord, 
                      color='black', marker='x', label='Points PV')
    axs[1, 0].scatter(loc_points_SV[0, :] / chord, loc_points_SV[1, :] / chord + 0.1034 / chord, 
                      color='black', marker='*', label='Points SV')
    # if len(loc_points_aux) > 0:
    #     axs[1, 0].scatter(loc_points_aux[0, :] / chord, loc_points_aux[1, :] / chord + 0.1034 / chord, 
    #                       color='black', marker='^', label='Points Aux')
    
    # --- Plot Strain Rate ---
    cmap = plt.cm.jet
    new_cmap = cmap(np.linspace(0, 1, 256))
    vmin, vmax = 0, 100
    lower_bound = 30
    n_colors = len(new_cmap)
    lower_idx = int((lower_bound - vmin) / (vmax - vmin) * n_colors)
    new_cmap[:lower_idx] = [1, 1, 1, 1]
    custom_cmap = mcolors.ListedColormap(new_cmap)
    
    cs5 = axs[1, 1].contourf(data_grid.grid_y / chord, 
                              data_grid.grid_z / chord + 0.1034 / chord, 
                              data_grid.mean_SR * chord / 30, levels=np.linspace(0, 100, 101), 
                              cmap=custom_cmap, extend='both')
    axs[1, 1].set_title('$S_{ij}S_{ij} c^2/U_{\infty}^2$')
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_xlabel('$y/c$')
    axs[1, 1].set_ylabel('$z/c$')
    
    # --- Plot Pressure Hessian ---
    cmap = plt.cm.jet
    new_cmap = cmap(np.linspace(0, 1, 256))
    vmin, vmax = -110000, 110000
    lower_bound, upper_bound = -20000, 20000
    n_colors = len(new_cmap)
    lower_idx = int((lower_bound - vmin) / (vmax - vmin) * n_colors)
    upper_idx = int((upper_bound - vmin) / (vmax - vmin) * n_colors)
    new_cmap[lower_idx:upper_idx] = [1, 1, 1, 1]
    custom_cmap = mcolors.ListedColormap(new_cmap)
    
    cs6 = axs[1, 2].contourf(data_grid.grid_y / chord, 
                              data_grid.grid_z / chord + 0.1034 / chord, 
                              data_grid.Pressure_Hessian/(0.5*30**2*0.305), 
                              levels=np.linspace(-10000, 10000, 201), 
                              cmap=custom_cmap, extend='both')
    axs[1, 2].set_title(r'${\nabla}^2 p / (1/2 \rho U_{\infty}^2)$')
    axs[1, 2].set_aspect('equal')
    axs[1, 2].set_xlabel('$y/c$')
    axs[1, 2].set_ylabel('$z/c$')
    
    # Add colorbars


    for ax, cs in zip(axs.flat, [cs1, cs2, cs3, cs4, cs5, cs6]):
        cbar = fig.colorbar(cs, ax=ax, orientation='vertical', pad=0.05)
        cbar.locator = MaxNLocator(nbins=5, prune='both')  # Set ~5 nice ticks, remove extremes
        cbar.update_ticks()
    
    fig.tight_layout()
    
    # Add airfoil mask for non-PIV3 locations
    if location != 'PIV3' and hasattr(data_grid, 'mask_indx') and data_grid.mask_indx is not None:
        mask = data_grid.mask_indx
        for ax in axs.flatten():
            ax.imshow(~mask, extent=(-0.05/chord, 0.05/chord, -0.15, 0.15), 
                      alpha=0.5, cmap='gray', aspect='auto')
    file_path = f'Velocity_Invariants_{location}_{data_type}'
    if limited_gradient:
        file_path += '_Limited'
    fig_name = os.path.join(file_path, f'Global_Velocity_Invariants_{location}.png')
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)

def plot_local_invariants_QR(location, R_hat, Q_hat, Vortex_Type: str, data_type, limited_gradient=False):
    """Plot local QR invariants in multi-panel subplot."""
    bins = 120
    num_points = int(np.shape(R_hat)[0])
    fig, axs = plt.subplots(int(num_points/2), 2, figsize=(int(num_points/2)*3+1, 8))
    axs = axs.flatten()
    
    for i in range(0, num_points):
        R_vals = R_hat[i, :]
        Q_vals = Q_hat[i, :]
        
        # Filter out NaN values before creating histogram
        valid_mask = ~(np.isnan(R_vals) | np.isnan(Q_vals))
        R_vals_clean = R_vals[valid_mask]
        Q_vals_clean = Q_vals[valid_mask]
        
        # Skip if no valid data
        if len(R_vals_clean) == 0:
            print(f"    Warning: No valid data for {Vortex_Type} QR plot")
            continue
            
        # Create 2D histogram
        h, edgesR, edgesQ = np.histogram2d(R_vals_clean, Q_vals_clean, bins=bins)
        totalPoints = R_vals.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Create centered meshgrid
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)
        
        # Set axis limits
        x_lim = max(np.abs(np.mean(R_vals) + 2 * np.std(R_vals)),
                    np.abs(np.mean(R_vals) - 2 * np.std(R_vals)))
        y_lim = max(np.abs(np.mean(Q_vals) + 2 * np.std(Q_vals)),
                    np.abs(np.mean(Q_vals) - 2 * np.std(Q_vals)))
        
        axs[i].set_xlim(-x_lim, x_lim)
        axs[i].set_ylim(-y_lim, y_lim)
        
        # Normalize and smooth PDF
        pdf_norm = pdf / np.max(pdf)
        pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
        
        # Plot contourså
        color_map = plt.cm.hot.reversed()
        c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                           cmap=color_map, extend='both')
        axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                       colors='blue', linestyles='dashed', linewidths=1)
        
        # Add discriminant curve
        x_zero_pdf = np.linspace(-x_lim, x_lim, 1000)
        Q_zero_pdf = -3 * ((x_zero_pdf / 2)**2)**(1/3)
        axs[i].axvline(0, color='black', linestyle='--', linewidth=2)
        axs[i].plot(x_zero_pdf, Q_zero_pdf, '--k', linewidth=2)
        
        axs[i].set_xlabel(r'$\hat{R}$', fontsize=14)
        axs[i].set_ylabel(r'$\hat{Q}$', fontsize=14)
        axs[i].set_title(f'Point {i+1}', fontsize=16)
        axs[i].tick_params(labelsize=12)
    
    fig.tight_layout()
    file_path = f'Velocity_Invariants_{location}_{data_type}'
    if limited_gradient:
        file_path += '_Limited'
    fig_name = os.path.join(file_path, f'Local_Velocity_Invariants_QR_{location}_{Vortex_Type}.png')
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)

def plot_local_invariants_Qs_Rs(location, Rs_hat, Qs_hat, Vortex_Type: str, data_type, limited_gradient=False):
    """Plot local strain rate invariants."""
    bins = 100
    a_values = [-0.5, 0, 1/2, 1/3, 1]
    num_points = int(np.shape(Rs_hat)[0])
    fig, axs = plt.subplots(int(num_points/2), 2, figsize=(int(num_points/2)*3+1, 8))
    axs = axs.flatten()
    
    for i in range(0, num_points):
        R_vals = Rs_hat[i, :]
        Q_vals = Qs_hat[i, :]
        
        # Filter out NaN values before creating histogram
        valid_mask = ~(np.isnan(R_vals) | np.isnan(Q_vals))
        R_vals_clean = R_vals[valid_mask]
        Q_vals_clean = Q_vals[valid_mask]
        
        # Skip if no valid data
        if len(R_vals_clean) == 0:
            print(f"    Warning: No valid data for {Vortex_Type} plot")
            continue
            
        # Create 2D histogram
        h, edgesR, edgesQ = np.histogram2d(R_vals_clean, Q_vals_clean, bins=bins)
        totalPoints = R_vals.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Create centered meshgrid
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)
        
        # Set axis limits
        x_lim = max(np.abs(np.mean(R_vals) + 2 * np.std(R_vals)),
                    np.abs(np.mean(R_vals) - 2 * np.std(R_vals)))
        y_lim = max(np.abs(np.mean(Q_vals) + 2 * np.std(Q_vals)),
                    np.abs(np.mean(Q_vals) - 2 * np.std(Q_vals)))
        
        axs[i].set_xlim(-x_lim, x_lim)
        axs[i].set_ylim(-y_lim, 0)
        
        # Normalize and smooth PDF
        pdf_norm = pdf / np.max(pdf)
        pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
        
        # Plot theoretical curves
        Q_plus = np.linspace(0.001, y_lim, 1000)
        for a in a_values:
            RS_line = - (Q_plus**(3/2)) * a * (1+a) / ((1+a+a**2)**(3/2))
            axs[i].plot(-RS_line, -Q_plus, '--', linewidth=2)
        
        # Plot contours
        color_map = plt.cm.hot.reversed()
        c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                           cmap=color_map, extend='both')
        axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                       colors='blue', linestyles='dashed', linewidths=1)
        
        axs[i].grid(True, which='both', linestyle='--', alpha=0.5)
        axs[i].set_xlabel(r'$\hat{R}_s$', fontsize=14)
        axs[i].set_ylabel(r'$\hat{Q}_s$', fontsize=14)
        axs[i].set_title(f'Point {i+1}', fontsize=16)
        axs[i].tick_params(labelsize=12)
    
    fig.tight_layout()
    file_path = f'Velocity_Invariants_{location}_{data_type}'
    if limited_gradient:
        file_path += '_Limited'
    fig_name = os.path.join(file_path, f'Local_Velocity_Invariants_Qs_Rs_{location}_{Vortex_Type}.png')
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)

def plot_local_invariants_Qs_Qw(location, Qw_hat, Qs_hat, Vortex_Type: str, data_type, limited_gradient=False):
    """Plot local strain-vorticity invariants."""
    bins = 100
    num_points = int(np.shape(Qs_hat)[0])
    fig, axs = plt.subplots(int(num_points/2), 2, figsize=(int(num_points/2)*3+1, 8))
    axs = axs.flatten()
    
    for i in range(0, num_points):
        R_vals = Qw_hat[i, :]
        Q_vals = -Qs_hat[i, :]
        
        # Filter out NaN values before creating histogram
        valid_mask = ~(np.isnan(R_vals) | np.isnan(Q_vals))
        R_vals_clean = R_vals[valid_mask]
        Q_vals_clean = Q_vals[valid_mask]
        
        # Skip if no valid data
        if len(R_vals_clean) == 0:
            print(f"    Warning: No valid data for {Vortex_Type} plot")
            continue
            
        # Create 2D histogram
        h, edgesR, edgesQ = np.histogram2d(R_vals_clean, Q_vals_clean, bins=bins)
        totalPoints = R_vals.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Create centered meshgrid
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)
        
        # Set axis limits
        x_lim = max(np.abs(np.mean(R_vals) + 2 * np.std(R_vals)),
                    np.abs(np.mean(R_vals) - 2 * np.std(R_vals)))
        y_lim = max(np.abs(np.mean(Q_vals) + 2 * np.std(Q_vals)),
                    np.abs(np.mean(Q_vals) - 2 * np.std(Q_vals)))
        
        axs[i].set_xlim(0, x_lim)
        axs[i].set_ylim(0, y_lim)
        
        # Normalize and smooth PDF
        pdf_norm = pdf / np.max(pdf)
        pdf_norm = gaussian_filter(pdf_norm, sigma=[1.5, 1.5])
        
        # Plot contours
        x_zero_pdf = np.linspace(x_pdf[0, 0], x_pdf[0, -1], 1000)
        Q_zero_pdf = x_zero_pdf
        cs = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                             cmap=plt.cm.hot.reversed(), extend='both')
        axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2],
                       colors='blue', linestyles='dashed', linewidths=1)
        
        # Add colorbar
        cbar = plt.colorbar(cs, ax=axs[i])
        cbar.set_ticks(np.linspace(0, 0.3, 4))
        
        axs[i].axvline(0, color='black', linestyle='--', linewidth=2)
        axs[i].plot(x_zero_pdf, Q_zero_pdf, '--k', linewidth=2)
        axs[i].set_xlabel(r'$\hat{Q}_w$', fontsize=14)
        axs[i].set_ylabel(r'$-\hat{Q}_s$', fontsize=14)
        axs[i].set_title(f'Point {i+1}', fontsize=16)
        axs[i].tick_params(labelsize=12)
    
    fig.tight_layout()
    file_path = f'Velocity_Invariants_{location}_{data_type}'
    if limited_gradient:
        file_path += '_Limited'
    fig_name = os.path.join(file_path, f'Local_Velocity_Invariants_Qs_Qw_{location}_{Vortex_Type}.png')
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)

def plot_vortex_profiles(vortex_type, location, grid, connectivity, P_Vortex, S_Vortex,
                         Qhat_all, Rhat_all, Qs_all, Rs_all, Qw_all, var_S, 
                         num_query_points: int = 100, L: float = 0.015, data_type: str = 'LES', limited_gradient=False):
    """
    Plot vortex profiles along the PCA axis for primary or secondary vortex.
    
    Parameters:
    -----------
    vortex_type : str
        'primary' or 'secondary'
    location : str
        Location identifier
    grid : tuple
        Grid coordinates (grid_x, grid_y)
    connectivity : array
        Connectivity matrix
    P_Vortex : object
        Primary vortex object
    S_Vortex : object
        Secondary vortex object
    Qhat_all, Rhat_all, Qs_all, Rs_all, Qw_all : array
        Invariant arrays with shape (N, T)
    var_S : array
        Mean S values for normalization
    num_query_points : int
        Number of query points along PCA axis
    L : float
        Maximum distance from starting point
    data_type : str
        Data type ('LES' or 'PIV')
    """
    # Load vortex core points from file
    output = f'Vortex_Detect_Results_{location}_{data_type}'
    
    if vortex_type.lower() == 'primary':
        core_file = f'{output}/P_core_{location}.npy'
        starting_point = P_Vortex.core.core_loc[0]
    elif vortex_type.lower() == 'secondary':
        core_file = f'{output}/S_core_{location}.npy'
        starting_point = S_Vortex.core.core_loc[0]
    else:
        raise ValueError("vortex_type must be 'primary' or 'secondary'")
    
    # Load core points for PCA
    core_points = np.load(core_file)
    
    # Compute PCA axis
    P_mean = np.mean(core_points, axis=0)
    pca = PCA(n_components=2)
    pca.fit(core_points - P_mean)
    pca_axis = pca.components_[0]
    print("            PCA axis:", pca_axis)
    
    # Define query points along PCA axis
    t_values = np.linspace(0, L, num_query_points)
    query_points = np.array([starting_point + t * pca_axis for t in t_values])
    
    # Extract profiles along PCA axis
    Q_profile_list = []
    R_profile_list = []
    Qs_profile_list = []
    Rs_profile_list = []
    Qw_profile_list = []
    
    for qp in query_points:
        # Find closest grid point
        dists = np.sqrt((grid[0] - qp[0])**2 + (grid[1] - qp[1])**2)
        idx = np.argmin(dists)
        
        # Find adjacent grid indices using connectivity
        rows = np.where(np.any(connectivity == idx, axis=1))[0]
        if len(rows) == 0:
            adj = np.array([idx])
        else:
            chosen_row = connectivity[rows[0], :]
            adj = np.unique(chosen_row)
        
        # Average over adjacent grid points
        Q_val = np.mean(Qhat_all[adj, :])
        R_val = np.mean(Rhat_all[adj, :])
        S_val = np.mean(var_S[adj])
        Qs_val = np.mean(Qs_all[adj, :]) / S_val
        Rs_val = np.mean(Rs_all[adj, :]) / (S_val**(3/2))
        Qw_val = np.mean(Qw_all[adj, :]) / S_val
        
        Q_profile_list.append(Q_val)
        R_profile_list.append(R_val)
        Qs_profile_list.append(Qs_val)
        Rs_profile_list.append(Rs_val)
        Qw_profile_list.append(Qw_val)
    
    # Convert to numpy arrays
    Q_profile = np.array(Q_profile_list)
    R_profile = np.array(R_profile_list)
    Qs_profile = np.array(Qs_profile_list)
    Rs_profile = np.array(Rs_profile_list)
    Qw_profile = np.array(Qw_profile_list)
    
    # Create plots
    a_values = [-0.5, 0, 1/2, 1/3, 1]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # QR plot
    ax1 = axs[0]
    ax1.plot(R_profile, Q_profile, 'o-', color='blue', label='Profile')
    ax1.plot(R_profile[0], Q_profile[0], 'ks', markersize=8, label='Core')
    ax1.plot(R_profile[-1], Q_profile[-1], 'k^', markersize=8, label='End')
    ax1.set_xlabel('$\hat{R}$')
    ax1.set_ylabel('$\hat{Q}$')
    
    # Set axis limits
    x_lim = max(max(R_profile)*1.2, abs(min(R_profile)*1.2))
    y_lim = max(max(Q_profile)*1.2, abs(min(Q_profile)*1.2))
    ax1.set_xlim(-x_lim, x_lim)
    ax1.set_ylim(-y_lim, y_lim)
    
    # Add discriminant curve
    x_zero = np.linspace(-x_lim, x_lim, 1000)
    Q_zero = -3 * ((x_zero / 2)**2)**(1/3)
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    ax1.plot(x_zero, Q_zero, '--k', linewidth=2)
    ax1.set_title('QR Plot along PCA Axis', fontsize=16)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()

    # Qs-Rs plot
    ax2 = axs[1]
    ax2.plot(Rs_profile, Qs_profile, 'o-', color='green', label='Profile')
    ax2.plot(Rs_profile[0], Qs_profile[0], 'ks', markersize=8, label='Core')
    ax2.plot(Rs_profile[-1], Qs_profile[-1], 'k^', markersize=8, label='End')
    
    # Set axis limits
    x_lim = max(max(Rs_profile)*1.2, abs(min(Rs_profile)*1.2))
    y_lim = max(max(Qs_profile)*1.2, abs(min(Qs_profile)*1.2))
    ax2.set_xlim(-x_lim, x_lim)
    ax2.set_ylim(-y_lim, 0)
    ax2.set_xlabel('$\hat{R}_s$')
    ax2.set_ylabel('$\hat{Q}_s$')
    ax2.set_title('Qs vs. Rs Plot along PCA Axis', fontsize=16)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Add theoretical curves
    Q_plus = np.linspace(0.001, y_lim, 1000)
    for a in a_values:
        RS_line = - (Q_plus**(3/2)) * a * (1 + a) / ((1 + a + a**2)**(3/2))
        ax2.plot(-RS_line, -Q_plus, '--', linewidth=2)
    ax2.legend()

    # Qs-Qw plot
    ax3 = axs[2]
    ax3.plot(Qw_profile, -Qs_profile, 'o-', color='red', label='Profile')
    ax3.plot(Qw_profile[0], -Qs_profile[0], 'ks', markersize=8, label='Core')
    ax3.plot(Qw_profile[-1], -Qs_profile[-1], 'k^', markersize=8, label='End')
    
    # Set axis limits
    x_lim = max(max(Qw_profile)*1.2, abs(min(Qw_profile)*1.2))
    y_lim = max(max(Qs_profile)*1.2, abs(min(Qs_profile)*1.2))
    ax3.set_xlim(0, x_lim)
    ax3.set_ylim(0, y_lim)
    ax3.axvline(0, color='black', linestyle='--', linewidth=2)
    ax3.plot(np.linspace(0.001, y_lim, 1000), np.linspace(0.001, y_lim, 1000), '--k', linewidth=2)
    ax3.set_xlabel('$\hat{Q}_w$')
    ax3.set_ylabel('-$\hat{Q}_s$')
    ax3.set_title('Qs vs. Qw Plot along PCA Axis', fontsize=16)
    ax3.grid(True, which='both', linestyle='--', alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    file_path = f'Velocity_Invariants_{location}_{data_type}'
    if limited_gradient:
        file_path += '_Limited'
    fig_name = os.path.join(file_path, f'Local_Velocity_Invariants_PCA_{location}_{vortex_type}.png')
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)

def plot_spectra(file_path: str,location: str, data_type: str, vortices=("PV","SV","TV","SS_shear","PS_shear"),
    point_indices=None,nchunk:int=8,# shared / per-panel x-limits
    xlim=None,xlim_vel=None,xlim_vort=None,xlim_p=None,xlim_tke=None):
    
    needed = ("u","v","w","vort_x","pressure")
    with h5py.File(file_path, "r") as f:
        if location not in f:
            raise KeyError(f"Location '{location}' not found. Available: {list(f.keys())}")
        g_loc = f[location]

        present_tags = [tag for tag in vortices if tag in g_loc]
        if not present_tags:
            raise KeyError(f"No requested vortices at '{location}'. Found: {list(g_loc.keys())}")

        V = {}
        for tag in present_tags:
            g = g_loc[tag]
            if not all(k in g for k in needed):
                continue
            if "dt" not in g.attrs:
                raise KeyError(f"Missing 'dt' attribute under {location}/{tag}")
            dt = float(g.attrs["dt"])
            arrs = {k: np.asarray(g[k]) for k in needed}
            npts = arrs["u"].shape[0] if arrs["u"].ndim == 2 else 1
            V[tag] = dict(**arrs, dt=dt, npts=npts)

        if not V:
            raise RuntimeError(f"No usable vortex tags at '{location}' (missing channels or dt).")

        n_points = int(min([V[tag]["npts"] for tag in V]))
        idx_list = list(range(n_points)) if point_indices is None else [i for i in np.atleast_1d(point_indices) if 0 <= i < n_points]
        if not idx_list:
            raise ValueError("point_indices resulted in an empty selection.")

        styles = {
            "PV":       dict(ls="-",  lw=1.8),
            "SV":       dict(ls="--", lw=1.8),
            "TV":       dict(ls=":",  lw=1.8),
            "SS_shear": dict(ls="-.", lw=1.6),
            "PS_shear": dict(ls=(0,(3,1)), lw=1.6),
        }

        for i in idx_list:
            psd = {}
            for tag, rec in V.items():
                def _row(a): return a[i, :] if a.ndim == 2 else a.ravel()
                u = _row(rec["u"]); v = _row(rec["v"]); w = _row(rec["w"])
                vx = _row(rec["vort_x"]); p = _row(rec["pressure"])
                dt = rec["dt"]

                f_u,  Su, = _welch_psd(u,  dt, nchunk=nchunk)
                f_v,  Sv = _welch_psd(v,  dt, nchunk=nchunk)
                f_w,  Sw = _welch_psd(w,  dt, nchunk=nchunk)
                f_vx, Svx = _welch_psd(vx, dt, nchunk=nchunk)
                f_p,  Sp = _welch_psd(p,  dt, nchunk=nchunk)

                Stke = 0.5 * (Su + Sv + Sw)         # linear TKE PSD
                ref2 = (2e-5)**2                     # pressure reference
                Sp_db = 10.0 * np.log10(np.maximum(Sp / ref2, 1e-300))

                psd[tag] = dict(
                    f=f_u, Su=Su, Sv=Sv, Sw=Sw,
                    fvx=f_vx, Svx=Svx,
                    fk=f_u, Stke=Stke,
                    fp=f_p, Sp_db=Sp_db
                )

            # ---------- 2 rows × 3 columns layout ----------
            fig, axs = plt.subplots(2, 3, figsize=(16.5, 8.2), sharex=True)
            fig.suptitle(f"{location} | point {i}", y=0.98)

            axU, axV, axW = axs[0,0], axs[0,1], axs[0,2]
            axP, axVX, axTKE = axs[1,0], axs[1,1], axs[1,2]

            # U, V, W (row 1)
            for tag, d in psd.items():
                axU.loglog(d["f"], d["Su"], **styles.get(tag, {}), label=tag, rasterized=True)
                axV.loglog(d["f"], d["Sv"], **styles.get(tag, {}), label=tag, rasterized=True)
                axW.loglog(d["f"], d["Sw"], **styles.get(tag, {}), label=tag, rasterized=True)
            axU.set_title("U"); axU.set_ylabel("PSD [m$^2$s$^{-2}$/Hz]")
            axV.set_title("V")
            axW.set_title("W")

            # P, ωx, TKE (row 2)
            for tag, d in psd.items():
                axP.semilogx(d["fp"], d["Sp_db"], **styles.get(tag, {}), label=tag, rasterized=True)
                axVX.loglog(d["fvx"], d["Svx"], **styles.get(tag, {}), label=tag, rasterized=True)
                axTKE.loglog(d["fk"], d["Stke"], **styles.get(tag, {}), label=tag, rasterized=True)
            axP.set_title("Pressure (ref 2e-5 Pa)"); axP.set_ylabel("Level [dB, ref 2e-5 Pa]")
            axVX.set_title(r"$\omega_x$");      axVX.set_ylabel("PSD [s$^{-2}$/Hz]")
            axTKE.set_title("TKE");             axTKE.set_ylabel("PSD [m$^2$s$^{-2}$/Hz]")

            # common x-labels (bottom row)
            for ax in axs[1,:]:
                ax.set_xlabel("f [Hz]")

            # x-limit handling
            def _sanitize_xlim(xl, default_min):
                if xl is None: return None
                lo, hi = xl
                if lo is None or lo <= 0: lo = default_min
                if (hi is not None) and (hi <= lo): hi = None
                return (lo, hi)

            all_pos = []
            for d in psd.values():
                for arr in (d["f"], d["fp"], d["fk"], d["fvx"]):
                    pos = arr[arr > 0]
                    if pos.size: all_pos.append(pos.min())
            min_pos = float(min(all_pos)) if all_pos else 1e-6

            xl_vel  = _sanitize_xlim(xlim_vel  if xlim_vel  is not None else xlim, min_pos)
            xl_vort = _sanitize_xlim(xlim_vort if xlim_vort is not None else xlim, min_pos)
            xl_p    = _sanitize_xlim(xlim_p    if xlim_p    is not None else xlim, min_pos)
            xl_tke  = _sanitize_xlim(xlim_tke  if xlim_tke  is not None else xlim, min_pos)

            if xl_vel:
                for ax in (axU, axV, axW): ax.set_xlim(xl_vel)
            if xl_p:    axP.set_xlim(xl_p)
            if xl_vort: axVX.set_xlim(xl_vort)
            if xl_tke:  axTKE.set_xlim(xl_tke)

            # grid + legend
            for ax in axs.flat:
                ax.grid(True, which="both", ls="--", alpha=0.35)
            # Single legend (bottom-right)
            handles, labels = axTKE.get_legend_handles_labels()
            if handles:
                axTKE.legend(loc="best", frameon=False, ncol=1)

            plt.tight_layout()
            file_path = f'Velocity_Invariants_{location}_{data_type}'
            fname = f"Local_Velocity_Spectra_{location}_pt{i:03d}.png"
            plt.savefig(os.path.join(file_path,fname), bbox_inches="tight", dpi=300)
            plt.close(fig)
