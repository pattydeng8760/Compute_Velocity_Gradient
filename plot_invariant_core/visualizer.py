import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from .utils import setup_rcparams, print
from .grid_maker import make_grid

def plot_global_invariants(data, connectivity, cut, data_type, vortex_detect_dir, 
                          velocity_invariant_dir, chord, velocity):
    """Plot global velocity invariants."""
    setup_rcparams()
    
    print(f"        Plotting global invariants for {cut}")
    
    # Create grid for plotting
    y_bnd = [-0.05, 0.05]
    z_bnd = [-0.16, -0.06]
    
    # Create interpolated grid
    data_grid = make_grid(500, y_bnd, z_bnd, data['y'], data['z'], 
                         np.mean(data['u'], axis=1), np.mean(data['vort_x'], axis=1), 'vort_x')
    
    # Add other variables to grid
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['Phat_all'], axis=1), 'Phat')
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['Qhat_all'], axis=1), 'Qhat')
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['Rhat_all'], axis=1), 'Rhat')
    data_grid.calculate_grid(500, y_bnd, z_bnd, np.mean(data['pressure_hessian'], axis=1), 'Pressure_Hessian')
    data_grid.calculate_grid(500, y_bnd, z_bnd, data['mean_SR'], 'mean_SR')
    
    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    
    # Plot Phat
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
                            data_grid.Phat, levels=np.arange(-1, 1, 0.01), 
                            cmap=custom_cmap, extend='both')
    axs[0, 0].set_title('$\hat{P}$')
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_xlabel('$y/c$')
    axs[0, 0].set_ylabel('$z/c$')
    
    # Plot Qhat
    bwr_cmap = plt.cm.bwr(np.linspace(0, 1, 256))
    new_colors = np.vstack(([1, 1, 1, 1], bwr_cmap, [1, 0, 0, 1]))
    custom_cmap = mcolors.ListedColormap(new_colors)
    
    cs2 = axs[0, 1].contourf(data_grid.grid_y / chord, 
                            data_grid.grid_z / chord + 0.1034 / chord, 
                            data_grid.Qhat, levels=np.arange(-1, 1, 0.01), 
                            cmap=custom_cmap, extend='both')
    axs[0, 1].set_title('$\hat{Q}$')
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_xlabel('$y/c$')
    axs[0, 1].set_ylabel('$z/c$')
    
    # Plot Rhat
    cs3 = axs[0, 2].contourf(data_grid.grid_y / chord, 
                            data_grid.grid_z / chord + 0.1034 / chord, 
                            data_grid.Rhat, levels=np.arange(-0.25, 0.25, 0.01), 
                            cmap=custom_cmap, extend='both')
    axs[0, 2].set_title('$\hat{R}$')
    axs[0, 2].set_aspect('equal')
    axs[0, 2].set_xlabel('$y/c$')
    axs[0, 2].set_ylabel('$z/c$')
    
    # Plot Vorticity
    cmap = plt.cm.bwr
    new_cmap = cmap(np.linspace(0, 1, 256))
    vmin, vmax = -120, 120
    lower_bound, upper_bound = -20, 20
    n_colors = len(new_cmap)
    lower_idx = int((lower_bound - vmin) / (vmax - vmin) * n_colors)
    upper_idx = int((upper_bound - vmin) / (vmax - vmin) * n_colors)
    new_cmap[lower_idx:upper_idx] = [1, 1, 1, 1]
    custom_cmap = mcolors.ListedColormap(new_cmap)
    
    cs4 = axs[1, 0].contourf(data_grid.grid_y / chord, 
                            data_grid.grid_z / chord + 0.1034 / chord, 
                            data_grid.vort_x * chord / velocity, levels=np.arange(-120, 120, 5), 
                            cmap=custom_cmap, extend='both')
    axs[1, 0].set_title('$\Omega_x c/U_{\infty}$')
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_xlabel('$y/c$')
    axs[1, 0].set_ylabel('$z/c$')
    
    # Plot Strain Rate
    cmap = plt.cm.jet
    new_cmap = cmap(np.linspace(0, 1, 256))
    vmin, vmax = 0, 150
    lower_bound = 50
    n_colors = len(new_cmap)
    lower_idx = int((lower_bound - vmin) / (vmax - vmin) * n_colors)
    new_cmap[:lower_idx] = [1, 1, 1, 1]
    custom_cmap = mcolors.ListedColormap(new_cmap)
    
    cs5 = axs[1, 1].contourf(data_grid.grid_y / chord, 
                            data_grid.grid_z / chord + 0.1034 / chord, 
                            data_grid.mean_SR * chord**2 / velocity**2, levels=np.arange(0, 150, 1), 
                            cmap=custom_cmap, extend='both')
    axs[1, 1].set_title('$S_{ij}S_{ij} c^2/U_{\infty}^2$')
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_xlabel('$y/c$')
    axs[1, 1].set_ylabel('$z/c$')
    
    # Plot Pressure Hessian
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
                            data_grid.Pressure_Hessian/(0.5*velocity**2*chord), 
                            levels=np.arange(-10000, 10000, 100), 
                            cmap=custom_cmap, extend='both')
    axs[1, 2].set_title(r'${\nabla}^2 p / (1/2 \rho U_{\infty}^2)$')
    axs[1, 2].set_aspect('equal')
    axs[1, 2].set_xlabel('$y/c$')
    axs[1, 2].set_ylabel('$z/c$')
    
    # Add colorbars
    for ax, cs in zip(axs.flat, [cs1, cs2, cs3, cs4, cs5, cs6]):
        fig.colorbar(cs, ax=ax, orientation='vertical', pad=0.05)
    
    fig.tight_layout()
    
    # Apply airfoil mask if applicable
    if cut != 'PIV3' and hasattr(data_grid, 'mask_indx'):
        mask = data_grid.mask_indx
        for ax in axs.flatten():
            ax.imshow(~mask, extent=(-0.05/chord, 0.05/chord, -0.15, 0.15), 
                     alpha=0.5, cmap='gray', aspect='auto')
    
    plt.savefig(f'Velocity_Invariants_{cut}_{data_type}/Global_Velocity_Invariants_{cut}.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_vortex_profiles(data, connectivity, cut, data_type, vortex_detect_dir, 
                        velocity_invariant_dir, chord, velocity):
    """Plot vortex profiles along PCA axes."""
    setup_rcparams()
    
    print(f"        Plotting vortex profiles for {cut}")
    
    # This would contain the PCA profile plotting logic
    # For now, just create a placeholder
    
    # Load vortex core data
    try:
        import os
        primary_cores = os.path.join(vortex_detect_dir, f'P_core_{cut}.npy')
        secondary_cores = os.path.join(vortex_detect_dir, f'S_core_{cut}.npy')
        
        if os.path.exists(primary_cores):
            plot_single_vortex_profile('primary', cut, data, connectivity, 
                                     primary_cores, data_type, chord, velocity)
        
        if os.path.exists(secondary_cores):
            plot_single_vortex_profile('secondary', cut, data, connectivity, 
                                     secondary_cores, data_type, chord, velocity)
    
    except Exception as e:
        print(f"        Error plotting vortex profiles: {e}")

def plot_single_vortex_profile(vortex_type, cut, data, connectivity, core_file, 
                             data_type, chord, velocity, num_query_points=100, L=0.015):
    """Plot profile for a single vortex along PCA axis."""
    # Load core points
    core_points = np.load(core_file)
    
    if len(core_points) == 0:
        print(f"            No core points found for {vortex_type} vortex")
        return
    
    # Compute PCA axis
    P_mean = np.mean(core_points, axis=0)
    pca = PCA(n_components=2)
    pca.fit(core_points - P_mean)
    pca_axis = pca.components_[0]
    
    # Define query points along PCA axis
    starting_point = core_points[0]  # Use first core point
    t_values = np.linspace(0, L, num_query_points)
    query_points = np.array([starting_point + t * pca_axis for t in t_values])
    
    # Extract profiles
    grid = [data['y'], data['z']]
    Q_profile = []
    R_profile = []
    Qs_profile = []
    Rs_profile = []
    Qw_profile = []
    
    for qp in query_points:
        # Find closest grid point
        dists = np.sqrt((grid[0] - qp[0])**2 + (grid[1] - qp[1])**2)
        idx = np.argmin(dists)
        
        # Use connectivity to find adjacent points
        rows = np.where(np.any(connectivity == idx, axis=1))[0]
        if len(rows) == 0:
            adj = np.array([idx])
        else:
            chosen_row = connectivity[rows[0], :]
            adj = np.unique(chosen_row)
        
        # Average over adjacent points
        Q_val = np.mean(data['Qhat_all'][adj, :])
        R_val = np.mean(data['Rhat_all'][adj, :])
        S_val = np.mean(data['var_S'][adj])
        Qs_val = np.mean(data['Qs_all'][adj, :]) / S_val
        Rs_val = np.mean(data['Rs_all'][adj, :]) / (S_val**(3/2))
        Qw_val = np.mean(data['Qw_all'][adj, :]) / S_val
        
        Q_profile.append(Q_val)
        R_profile.append(R_val)
        Qs_profile.append(Qs_val)
        Rs_profile.append(Rs_val)
        Qw_profile.append(Qw_val)
    
    # Convert to arrays
    Q_profile = np.array(Q_profile)
    R_profile = np.array(R_profile)
    Qs_profile = np.array(Qs_profile)
    Rs_profile = np.array(Rs_profile)
    Qw_profile = np.array(Qw_profile)
    
    # Create plots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # QR plot
    axs[0].plot(R_profile, Q_profile, 'o-', color='blue', label='Profile')
    axs[0].plot(R_profile[0], Q_profile[0], 'ks', markersize=8, label='Core')
    axs[0].plot(R_profile[-1], Q_profile[-1], 'k^', markersize=8, label='End')
    axs[0].set_xlabel('$\hat{R}$')
    axs[0].set_ylabel('$\hat{Q}$')
    axs[0].set_title('QR Plot along PCA Axis', fontsize=16)
    axs[0].grid(True, which='both', linestyle='--', alpha=0.5)
    axs[0].legend()
    
    # Add delta=0 curve
    x_lim = max(max(R_profile)*1.2, abs(min(R_profile)*1.2))
    y_lim = max(max(Q_profile)*1.2, abs(min(Q_profile)*1.2))
    axs[0].set_xlim(-x_lim, x_lim)
    axs[0].set_ylim(-y_lim, y_lim)
    x_zero = np.linspace(-x_lim, x_lim, 1000)
    Q_zero = -3 * ((x_zero / 2)**2)**(1/3)
    axs[0].axvline(0, color='black', linestyle='--', linewidth=2)
    axs[0].plot(x_zero, Q_zero, '--k', linewidth=2)
    
    # Qs-Rs plot
    axs[1].plot(Rs_profile, Qs_profile, 'o-', color='green', label='Profile')
    axs[1].plot(Rs_profile[0], Qs_profile[0], 'ks', markersize=8, label='Core')
    axs[1].plot(Rs_profile[-1], Qs_profile[-1], 'k^', markersize=8, label='End')
    axs[1].set_xlabel('$\hat{R}_s$')
    axs[1].set_ylabel('$\hat{Q}_s$')
    axs[1].set_title('Qs vs. Rs Plot along PCA Axis', fontsize=16)
    axs[1].grid(True, which='both', linestyle='--', alpha=0.5)
    axs[1].legend()
    
    # Add theoretical curves
    x_lim = max(max(Rs_profile)*1.2, abs(min(Rs_profile)*1.2))
    y_lim = max(max(Qs_profile)*1.2, abs(min(Qs_profile)*1.2))
    axs[1].set_xlim(-x_lim, x_lim)
    axs[1].set_ylim(-y_lim, 0)
    
    a_values = [-0.5, 0, 1/2, 1/3, 1]
    Q_plus = np.linspace(0.001, y_lim, 1000)
    for a in a_values:
        RS_line = - (Q_plus**(3/2)) * a * (1 + a) / ((1 + a + a**2)**(3/2))
        axs[1].plot(-RS_line, -Q_plus, '--', linewidth=2)
    
    # Qs-Qw plot
    axs[2].plot(Qw_profile, -Qs_profile, 'o-', color='red', label='Profile')
    axs[2].plot(Qw_profile[0], -Qs_profile[0], 'ks', markersize=8, label='Core')
    axs[2].plot(Qw_profile[-1], -Qs_profile[-1], 'k^', markersize=8, label='End')
    axs[2].set_xlabel('$\hat{Q}_w$')
    axs[2].set_ylabel('-$\hat{Q}_s$')
    axs[2].set_title('Qs vs. Qw Plot along PCA Axis', fontsize=16)
    axs[2].grid(True, which='both', linestyle='--', alpha=0.5)
    axs[2].legend()
    
    # Set limits and add reference line
    x_lim = max(max(Qw_profile)*1.2, abs(min(Qw_profile)*1.2))
    y_lim = max(max(Qs_profile)*1.2, abs(min(Qs_profile)*1.2))
    axs[2].set_xlim(0, x_lim)
    axs[2].set_ylim(0, y_lim)
    axs[2].axvline(0, color='black', linestyle='--', linewidth=2)
    axs[2].plot(np.linspace(0.001, y_lim, 1000), np.linspace(0.001, y_lim, 1000), '--k', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f'Velocity_Invariants_{cut}_{data_type}/Local_Velocity_Invariants_PCA_{cut}_{vortex_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"            Generated {vortex_type} vortex profile plot")