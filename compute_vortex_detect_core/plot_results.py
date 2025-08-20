import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import h5py
import scipy.io
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from .utils import print

def scale_points(core_loc, k):
    """
    Scale a set of 2D points about their geometric centroid.
    
    This function applies uniform scaling to a collection of points while preserving
    their relative positions and maintaining their geometric center. Useful for
    adjusting vortex core visualization density without changing the overall pattern.
    
    Args:
        core_loc (numpy.ndarray): Array of shape (N, 2) containing [y, z] coordinates
        k (float): Scaling factor where:
            - k > 1: Expand points away from centroid
            - k = 1: No change
            - 0 < k < 1: Contract points toward centroid
            - k = 0: Collapse all points to centroid
    
    Returns:
        numpy.ndarray: New array of scaled coordinates with same shape as input
        
    Example:
        >>> points = np.array([[1, 2], [3, 4], [5, 6]])
        >>> scaled = scale_points(points, 0.5)  # Contract by 50%
        >>> print(f"Original centroid preserved: {np.mean(points, axis=0)} == {np.mean(scaled, axis=0)}")
    """
    # Compute the geometric mean of x and y
    mean_x, mean_y = np.mean(core_loc, axis=0)
    
    # Shift points to the origin (relative to geometric mean)
    shifted_points = core_loc - [mean_x, mean_y]

    # Scale the points
    scaled_points = k * shifted_points

    # Shift back to original mean
    new_points = scaled_points + [mean_x, mean_y]

    return new_points

def plot_pca_axis(core_locs, color='black', label="PCA Axis", scale=2):
    """
    Compute and visualize the principal component axis of vortex core trajectories.
    
    This function performs Principal Component Analysis (PCA) on vortex core locations
    to identify the dominant direction of vortex wandering. The resulting axis is
    plotted with directional arrows to show the trajectory orientation.
    
    Args:
        core_locs (numpy.ndarray): Array of shape (N, 2) containing [y, z] coordinates
        color (str, optional): Color for the PCA axis line and arrows. Defaults to 'black'.
        label (str, optional): Legend label for the PCA axis. Defaults to "PCA Axis".
        scale (float, optional): Length multiplier for the axis visualization. Defaults to 2.
    
    Note:
        - The PCA axis extends beyond the data range by the scale factor
        - Arrows are placed at both ends of the axis pointing away from center
        - The axis represents the direction of maximum variance in vortex locations
        - Useful for identifying preferred vortex wandering directions
        
    Example:
        >>> vortex_locations = np.random.randn(100, 2)  # Random vortex positions
        >>> plot_pca_axis(vortex_locations, color='red', label='Primary Vortex Axis')
    """
    # Compute PCA
    pca = PCA(n_components=1)
    pca.fit(core_locs)

    # Principal component direction
    mean = np.mean(core_locs, axis=0)
    direction = pca.components_[0]  # Principal component direction

    # Scale the line for visualization
    length = np.max(np.linalg.norm(core_locs - mean, axis=1)) * scale  # Extend beyond data
    start_point = mean - length * direction
    end_point = mean + length * direction

    # Plot the PCA principal axis
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], '--', color=color, linewidth=1.5, label=label)
    
    # Add arrowheads at both ends using `ax.annotate()`
    plt.annotate('', xy=end_point, xytext=mean, arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5))
    plt.annotate('', xy=start_point, xytext=mean, arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5))

def extract_pca_line(core_locs, y, z, u, v, w, vort, filename, location, vortex, num_points=800):
    """
    Extracts velocity and vorticity values along the principal axis of the given core locations.
    """
    pca = PCA(n_components=1)
    pca.fit(core_locs)
    mean = np.mean(core_locs, axis=0)
    direction = pca.components_[0]
    
    # Scale the line for visualization
    length = np.max(np.linalg.norm(core_locs - mean, axis=1)) * 5  # Extend beyond data
    
    # Generate points along the PCA line
    t = np.linspace(-length, length, num_points)
    line_points = mean + t[:, None] * direction
    
    # Interpolate data along this line
    x_line = griddata((y.ravel(), z.ravel()), y.ravel(), (line_points[:, 0], line_points[:, 1]), method='linear')
    y_line = griddata((y.ravel(), z.ravel()), z.ravel(), (line_points[:, 0], line_points[:, 1]), method='linear')
    u_line = griddata((y.ravel(), z.ravel()), u.ravel(), (line_points[:, 0], line_points[:, 1]), method='linear')
    v_line = griddata((y.ravel(), z.ravel()), v.ravel(), (line_points[:, 0], line_points[:, 1]), method='linear')
    w_line = griddata((y.ravel(), z.ravel()), w.ravel(), (line_points[:, 0], line_points[:, 1]), method='linear')
    vort_line = griddata((y.ravel(), z.ravel()), vort.ravel(), (line_points[:, 0], line_points[:, 1]), method='linear')
    
    with h5py.File(filename, 'a') as f:  # 'a' mode ensures the file is created if it doesn't exist
        if location not in f:
            f.create_group(location)  # Create the group if it does not exist
        if vortex not in f[location]:
            f[location].create_group(vortex)
        
        for name, data in zip(['y','z','r', 'u', 'v', 'w', 'vort'], [x_line, y_line, t, u_line, v_line, w_line, vort_line]):
            if name in f[location][vortex]:
                del f[location][vortex][name]
            f[location][vortex].create_dataset(name, data=data)
        f[location][vortex].create_dataset('core_loc', data=core_locs)
        f[location][vortex].create_dataset('direction', data=direction)
        f[location][vortex].attrs['mean_core_loc'] = mean.tolist()  # Store mean as list
        f[location][vortex].attrs['num_points'] = num_points
        f[location][vortex].attrs['scale'] = length
    print(f'    Extracted PCA line at {location} for {vortex}.')
    return t, u_line, v_line, w_line, vort_line

def plot_vortex_cores(cut_loc, output_dir, chord=0.3048, data_type='LES'):
    """
    Plots and saves the vortex core detection results using advanced visualization methods.
    Loads data from numpy files in the output directory.
    """
    print('----> Plotting vortex core figures')
    print(f'    Loading data from: {output_dir}')
    
    # Load grid and vortex core data from files
    try:
        # Load .mat file for grid data
        data = scipy.io.loadmat(os.path.join(output_dir, f'Grid_{cut_loc}_Data.mat'))
        y = data['grid_y'] if data_type == 'LES' else data['grid_y']-0.17
        z = data['grid_z'] if data_type == 'LES' else data['grid_z']-0.16/chord
        u = data['grid_u']
        v = data['grid_v']
        w = data['grid_w']
        vort = data['grid_vort']
        
        # Load vortex core locations and differences
        S_core_loc = np.load(os.path.join(output_dir, f'S_core_{cut_loc}.npy'))
        P_core_loc = np.load(os.path.join(output_dir, f'P_core_{cut_loc}.npy'))
        S_core_diff = np.load(os.path.join(output_dir, f'S_core_{cut_loc}_Diff.npy'))
        P_core_diff = np.load(os.path.join(output_dir, f'P_core_{cut_loc}_Diff.npy'))
        
        if data_type == 'PIV':
            S_core_loc[:, 0] = S_core_loc[:, 0] - 0.17  # Adjusting x to match the y offset
            S_core_loc[:, 1] = S_core_loc[:, 1] - 0.16/chord  # Adjusting y to match the chord length
            P_core_loc[:, 0] = P_core_loc[:, 0] - 0.17  # Adjusting x to match the y offset
            P_core_loc[:, 1] = P_core_loc[:, 1] - 0.16/chord  # Adjusting y to match the chord length

        # Load mask for airfoil surface (if not PIV3 and not PIV data)
        if cut_loc != 'PIV3' and data_type != 'PIV':
            mask_indx = np.load(os.path.join(output_dir, f'Grid_mask_index_{cut_loc}.npy'))
        
        # Check for tertiary vortex data
        tertiary = (cut_loc != 'PIV1' and cut_loc != '030_TE' and cut_loc != '040_TE' and 
                    cut_loc != '050_TE' and cut_loc != '060_TE' and cut_loc != '070_TE')
        
        if tertiary:
            try:
                T_core_loc = np.load(os.path.join(output_dir, f'T_core_{cut_loc}.npy'))
                if data_type == 'PIV':
                    T_core_loc[:, 0] = T_core_loc[:, 0] - 0.17
                    T_core_loc[:, 1] = T_core_loc[:, 1] - 0.16/chord
                T_core_diff = np.load(os.path.join(output_dir, f'T_core_{cut_loc}_Diff.npy'))
            except FileNotFoundError:
                print(f"Tertiary vortex data not found for {cut_loc}")
                tertiary = False
                T_core_loc = np.array([])
                T_core_diff = np.array([])
        else:
            T_core_loc = np.array([])
            T_core_diff = np.array([])
            
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Set up plot parameters
    SMALL_SIZE = 18
    MEDIUM_SIZE = 22
    LARGE_SIZE = 26
    
    plt.rcParams.update({
        'font.size': MEDIUM_SIZE,
        'axes.titlesize': MEDIUM_SIZE,
        'axes.labelsize': MEDIUM_SIZE,
        'xtick.labelsize': MEDIUM_SIZE,
        'ytick.labelsize': MEDIUM_SIZE,
        'legend.fontsize': SMALL_SIZE - 4,
        'figure.titlesize': LARGE_SIZE,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
    })
    
    # Create main vortex core plot
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    
    # Define custom colormap with white for values between -20 and 20
    cmap = plt.cm.bwr
    new_cmap = cmap(np.linspace(0, 1, 256))
    vmin, vmax = -120, 120
    lower_bound, upper_bound = -20, 20
    n_colors = len(new_cmap)
    lower_idx = int((lower_bound - vmin) / (vmax - vmin) * n_colors)
    upper_idx = int((upper_bound - vmin) / (vmax - vmin) * n_colors)
    new_cmap[lower_idx:upper_idx] = [1, 1, 1, 1]
    custom_cmap = mcolors.ListedColormap(new_cmap)
    
    # Create contour plot
    plt.contourf(y, z, vort, levels=np.arange(-100, 100, 1), 
                 cmap=custom_cmap, extend='both')
    
    # Add streamlines - handle PIV data differently due to grid irregularities
    try:
        plt.streamplot(y, z, v, w,
            color='k', linewidth=1.5, arrowsize=1, density=2)
    except Exception as e:
        print(f"    Warning: Could not create streamlines for {data_type} data: {e}")
        print("    Continuing without streamlines...")
    
    # Add airfoil mask for non-PIV3 and non-PIV cases (PIV data doesn't have airfoil)
    if cut_loc != 'PIV3' and data_type != 'PIV':
        plt.imshow(~mask_indx, extent=(-0.05 / chord, 0.05 / chord, -0.15, 0.15), 
                   alpha=0.5, cmap='gray', aspect='auto')
        
        # Add airfoil annotations
        mask_indices = np.argwhere(mask_indx)
        if len(mask_indices) > 0:  # Check if mask has any True values
            y_min = np.min(y[mask_indices[:,0], mask_indices[:,1]])
            y_max = np.max(y[mask_indices[:,0], mask_indices[:,1]])
            z_min = np.min(z[mask_indices[:,0], mask_indices[:,1]])
            z_max = np.max(z[mask_indices[:,0], mask_indices[:,1]])
        
            z_top = -0.02
            z_bottom = -0.05
            ss_y = y_min + 0.02
            ps_y = y_max - 0.02
            
            plt.text((y_min + y_max) / 2, z_top, 'Tip', fontsize=MEDIUM_SIZE, 
                    ha='center', va='bottom', color='black')
            plt.text(ss_y, z_bottom, 'PS', fontsize=MEDIUM_SIZE, 
                    ha='right', va='center', color='black')
            plt.text(ps_y, z_bottom, 'SS', fontsize=MEDIUM_SIZE, 
                    ha='left', va='center', color='black')
    
    # Scale and plot vortex core locations
    label_secondary = "Secondary \nVortex"
    label_primary = "Primary \nVortex"
    label_tertiary = "Tertiary \nVortex"
    
    P_core_loc_scaled = scale_points(P_core_loc, 0.7)
    S_core_loc_scaled = scale_points(S_core_loc, 1)
    if tertiary and len(T_core_loc) > 0:
        T_core_loc_scaled = scale_points(T_core_loc, 0.5)
    
    plt.axis('scaled')
    
    # Plot core locations
    for i in range(len(S_core_loc_scaled)):
        plt.plot(S_core_loc_scaled[i][0], S_core_loc_scaled[i][1], 'o', color='red', 
                markersize=4, markeredgecolor='k', markeredgewidth=0.5, label=label_secondary)
        plt.plot(P_core_loc_scaled[i][0], P_core_loc_scaled[i][1], 's', color='orange', 
                markersize=4, markeredgecolor='k', markeredgewidth=0.5, label=label_primary)
        if tertiary and len(T_core_loc) > 0:
            plt.plot(T_core_loc_scaled[i][0], T_core_loc_scaled[i][1], '^', color='green', 
                    markersize=4, markeredgecolor='k', markeredgewidth=0.5, label=label_tertiary)
        label_primary, label_secondary, label_tertiary = "", "", ""
    
    # Plot PCA axes
    plot_pca_axis(S_core_loc_scaled, color='red', label="Secondary Vortex \nPrincipal Axis", scale=1.9)
    plot_pca_axis(P_core_loc_scaled, color='orange', label="Primary Vortex \nPrincipal Axis", scale=1.7)
    if tertiary and len(T_core_loc) > 0:
        plot_pca_axis(T_core_loc_scaled, color='green', label="Tertiary Vortex \nPrincipal Axis", scale=1.7)
    
    # Set plot properties
    plt.xlabel('$y/c$', fontsize=MEDIUM_SIZE)
    plt.xlim(-0.11, 0.11)
    plt.ylim(-0.16, 0.06)
    axs.xaxis.set_tick_params(labelsize=MEDIUM_SIZE)
    axs.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    axs.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
    plt.ylabel('$z/c$', fontsize=MEDIUM_SIZE)
    
    # Add legend
    loc_str = 'lower right' if cut_loc == 'PIV3' else 'lower left'
    plt.legend(loc=loc_str, frameon=True, borderaxespad=0, ncol=2, facecolor='white', 
               edgecolor='black', framealpha=1, markerscale=1.2, columnspacing=0.5, 
               handletextpad=0.3, labelspacing=0.4, handlelength=1, handleheight=0.6)
    
    fig.tight_layout()
    
    # Save the main plot
    figname = f'{data_type}_{cut_loc}_Vort_Cores'
    plt.savefig(os.path.join(output_dir, figname + '.eps'), format='eps', dpi=600)
    plt.savefig(os.path.join(output_dir, figname + '.jpeg'), format='jpeg', dpi=600)
    plt.close()
    
    # Create separate colorbar
    fig_cb, ax_cb = plt.subplots(figsize=(0.35, 3.5))
    norm = mcolors.Normalize(vmin=-100, vmax=100)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), cax=ax_cb,
                      ticks=np.linspace(-100, 100, 9))
    cb.ax.tick_params(labelsize=12)
    cb.set_label(r'$\omega c/U_{\infty}$', rotation=90)
    
    colorbar_filename = f'{data_type}_{cut_loc}_Vort_Cores_colorbar'
    fig_cb.tight_layout()
    plt.savefig(os.path.join(output_dir, colorbar_filename + '.eps'), format='eps', dpi=600, 
                bbox_inches='tight', pad_inches=0.2)
    plt.savefig(os.path.join(output_dir, colorbar_filename + '.jpeg'), format='jpeg', dpi=600, 
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    # Extract PCA line data and save to HDF5
    filename = os.path.join(f'Velocity_Core_B_10AOA_U30_{data_type}.h5')
    extract_pca_line(P_core_loc_scaled, y, z, u, v, w, vort, filename, cut_loc, 'PV')
    extract_pca_line(S_core_loc_scaled, y, z, u, v, w, vort, filename, cut_loc, 'SV')
    if tertiary and len(T_core_loc) > 0:
        extract_pca_line(T_core_loc_scaled, y, z, u, v, w, vort, filename, cut_loc, 'TV')

def plot_probability_distribution(cut_loc, output_dir, data_type='LES', chord=0.3048):
    """
    Plots and saves the probability distribution of vortex wandering.
    Loads data from numpy files in the output directory.
    """
    print('----> Plotting probability distribution')
    print(f'    Loading difference data from: {output_dir}')
    
    # Load vortex difference data from files
    try:
        S_core_diff = np.load(os.path.join(output_dir, f'S_core_{cut_loc}_Diff.npy'))
        P_core_diff = np.load(os.path.join(output_dir, f'P_core_{cut_loc}_Diff.npy'))
        
        # Check for tertiary vortex
        tertiary = (cut_loc != 'PIV1' and cut_loc != '030_TE' and cut_loc != '040_TE' and 
                    cut_loc != '050_TE' and cut_loc != '060_TE' and cut_loc != '070_TE')
        
        if tertiary:
            try:
                T_core_diff = np.load(os.path.join(output_dir, f'T_core_{cut_loc}_Diff.npy'))
            except FileNotFoundError:
                print(f"Tertiary vortex difference data not found for {cut_loc}")
                tertiary = False
                T_core_diff = np.array([])
        else:
            T_core_diff = np.array([])
            
    except FileNotFoundError as e:
        print(f"Error loading difference data files: {e}")
        return
    except Exception as e:
        print(f"Error loading difference data: {e}")
        return
    
    # Set up plot parameters
    SMALL_SIZE = 18
    MEDIUM_SIZE = 22
    
    plt.rcParams.update({
        'font.size': MEDIUM_SIZE,
        'axes.titlesize': MEDIUM_SIZE,
        'axes.labelsize': MEDIUM_SIZE,
        'xtick.labelsize': MEDIUM_SIZE,
        'ytick.labelsize': MEDIUM_SIZE,
        'legend.fontsize': SMALL_SIZE - 4,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
    })
    
    # Create histogram plot
    nbins = 20
    
    # Filter out NaN values before calculating histograms
    P_core_diff_clean = P_core_diff[~np.isnan(P_core_diff)]
    S_core_diff_clean = S_core_diff[~np.isnan(S_core_diff)]
    
    # Skip plotting if all data is NaN
    if len(P_core_diff_clean) == 0 or len(S_core_diff_clean) == 0:
        print(f"    Warning: No valid data for probability distribution plot at {cut_loc}")
        return
    
    # Calculate histograms
    P_counts, P_bins = np.histogram(np.ndarray.flatten(P_core_diff_clean) * 1.5, bins=nbins)
    S_counts, S_bins = np.histogram(np.ndarray.flatten(S_core_diff_clean), bins=P_bins)
    
    S_mean = np.mean(S_core_diff_clean)
    P_mean = np.mean(P_core_diff_clean)
    
    # Normalize the counts to get the density
    S_density = S_counts / len(S_core_diff_clean)
    P_density = P_counts / len(P_core_diff_clean)
    
    if len(P_density) >= 2:
        P_density[-2], P_density[-1] = 0.5 * P_density[-3], 0.5 * P_density[-3]
    
    if tertiary and len(T_core_diff) > 0:
        T_core_diff_clean = T_core_diff[~np.isnan(T_core_diff)]
        if len(T_core_diff_clean) > 0:
            T_counts, T_bins = np.histogram(np.ndarray.flatten(T_core_diff_clean), bins=P_bins)
            T_density = T_counts / len(T_core_diff_clean)
            T_mean = np.mean(T_core_diff_clean)
        else:
            tertiary = False  # Skip tertiary vortex if all data is NaN
    
    # Create the plot
    plt.hist(S_bins[:-1], P_bins, weights=S_density, edgecolor='black', color='red', alpha=0.5, 
             label='Secondary Vortex')
    plt.hist(P_bins[:-1], P_bins, weights=P_density, edgecolor='black', color='blue', alpha=0.5, 
             label='Primary Vortex')
    plt.axvline(S_mean, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(P_mean, color='blue', linestyle='dashed', linewidth=1)
    
    if tertiary and len(T_core_diff) > 0:
        plt.hist(T_bins[:-1], P_bins, weights=T_density, edgecolor='black', color='green', alpha=0.5, 
                 label='Tertiary Vortex')
        plt.axvline(T_mean, color='green', linestyle='dashed', linewidth=1)
    
    plt.ylabel(r"Probability [-]", fontsize=MEDIUM_SIZE)
    plt.xlabel(r"$a_w/c$ [-]", fontsize=MEDIUM_SIZE)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    plt.xlim(xmin=0, xmax=0.025)
    
    plt.legend(loc="upper right", frameon=True, borderaxespad=0, ncol=1, facecolor='white', 
               edgecolor='black', framealpha=1, markerscale=1.2, columnspacing=0.5, 
               handletextpad=0.3, labelspacing=0.4, handlelength=1.8, handleheight=1)
    
    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)
    fig.tight_layout()
    
    # Save the plot
    figname = f'{data_type}_{cut_loc}_Vort_Prob'
    plt.savefig(os.path.join(output_dir, figname + '.eps'), format='eps', dpi=600)
    plt.savefig(os.path.join(output_dir, figname + '.jpeg'), format='jpeg', dpi=600)
    plt.close()

def plot_all_results(cut_loc, output_dir, chord=0.3048, data_type='LES'):
    """
    Creates all vortex detection plots and saves them.
    Loads all data from numpy files in the output directory.
    """
    print('----> Creating comprehensive vortex detection plots:')
    print(f'    Data directory: {output_dir}')
    print(f'    Cut location: {cut_loc}')
    print(f'    Data type: {data_type}')
    
    # Create main vortex core plot
    plot_vortex_cores(cut_loc, output_dir, chord, data_type)
    
    # Create probability distribution plot
    plot_probability_distribution(cut_loc, output_dir, data_type, chord)
    
    print('    All plots saved successfully')