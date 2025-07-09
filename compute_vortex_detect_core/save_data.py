import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging


def save_data(vars,cut,P_core_loc,P_Vort_Diff,S_core_loc,S_Vort_Diff,T_core_loc,T_Vort_Diff,dir,tertiary=False):
    """
    Saves the vortex detection results.
    """
    logger.info('Saving data files')

    # Saving the grid data for plotting
    chord = 0.3048
    # Saving the numpy file
    np.save(os.path.join(dir, f'Grid_y_{cut}'), vars.grid_y / chord)
    np.save(os.path.join(dir, f'Grid_z_{cut}'), vars.grid_z / chord + 0.1034/chord)
    np.save(os.path.join(dir, f'Grid_vort_{cut}'), vars.grid_vort)
    np.save(os.path.join(dir, f'Grid_u_{cut}'), vars.grid_u)
    np.save(os.path.join(dir, f'Grid_v_{cut}'), vars.grid_v)
    np.save(os.path.join(dir, f'Grid_w_{cut}'), vars.grid_w)
    # Save the data as matlab file
    scipy.io.savemat(os.path.join(dir, f'Grid_{cut}_Data.mat'), 
                     {'grid_y': vars.grid_y/chord, 'grid_z': vars.grid_z/ chord + 0.1034/chord, 'grid_vort': vars.grid_vort, 'grid_u': vars.grid_u ,'grid_v': vars.grid_v, 'grid_w': vars.grid_w})
    if tertiary== False:
        np.save(os.path.join(dir, f'Grid_mask_index_{cut}'), vars.mask_indx)
    P_core_loc[:,0] = P_core_loc[:,0]/chord
    P_core_loc[:,1] = P_core_loc[:,1]/chord + 0.1034/chord
    S_core_loc[:,0] = S_core_loc[:,0]/chord
    S_core_loc[:,1] = S_core_loc[:,1]/chord + 0.1034/chord
    # Saving the vortex core locations and trace differences
    np.save(os.path.join(dir, f'S_core_{cut}'), S_core_loc)
    np.save(os.path.join(dir, f'P_core_{cut}'), P_core_loc)
    np.save(os.path.join(dir, f'S_core_{cut}_Diff'), S_Vort_Diff.diff)
    np.save(os.path.join(dir, f'P_core_{cut}_Diff'), P_Vort_Diff.diff)
    # Saving the output as hdf5 file
    # output_file = os.path.join(dir, 'Vortex_Core_' + cut + '.h5')
    # with h5py.File(output_file, 'w') as f:
    #     f.create_dataset('y', grid_y_array, dtype='float32')
    #     f.create_dataset('z', grid_z_array, dtype='float32')
    #     f.create_dataset('vort', grid_vort_array, dtype='float32')
    #     f.create_dataset('u', grid_u_array,dtype='float32') 
    #     f.create_dataset('v', grid_v_array,dtype='float32')
    #     f.create_dataset('w', grid_w_array,dtype='float32')
    #     if cut != 'PIV3':
    #         f.create_dataset('mask_indx', mask_indx_array)
    #     if cut != 'PIV1':
    #         f.create_dataset('T_core', np.array(T_core_loc),dtype='float32')
    #         f.create_dataset('T_core_diff', np.array(T_Vort_Diff.diff),dtype='float32')
    if tertiary:
        T_core_loc[:,0] = T_core_loc[:,0]/chord
        T_core_loc[:,1] = T_core_loc[:,1]/chord + 0.1034/chord  
        np.save(os.path.join(dir, f'T_core_{cut}'), T_core_loc)
        np.save(os.path.join(dir, f'T_core_{cut}_Diff'), T_Vort_Diff.diff)


def Plot_Result(S_core_loc, S_Vort_Diff, P_core_loc, P_Vort_Diff, T_core_loc, T_Vort_Diff, Vars, cut_loc, dir, chord=0.3048):
    """
    Plots and saves the vortex detection results.
    """
    logger.info('Plotting figures')
    ## Plotting contour
    # The overall plot size
    SMALL_SIZE = 12
    MEDIUM_SIZE = 18
    LARGE_SIZE = 22
    fig, axs = plt.subplots(1, 1)
    plt.rcParams.update({
        'font.size': MEDIUM_SIZE,           # Default text sizes
        'axes.titlesize': MEDIUM_SIZE,      # Axes title font size
        'axes.labelsize': MEDIUM_SIZE + 2,  # X and Y labels font size
        'xtick.labelsize': MEDIUM_SIZE - 2, # X tick labels font size
        'ytick.labelsize': MEDIUM_SIZE - 2, # Y tick labels font size
        'legend.fontsize': SMALL_SIZE - 4,  # Legend font size
        'figure.titlesize': LARGE_SIZE,     # Figure title font size
    })
    
    plt.contourf(Vars.grid_y / chord, Vars.grid_z / chord + 0.1034 / chord, Vars.grid_vort, levels=np.arange(-100, 100, 1), cmap='RdBu', extend='both')
    cb = plt.colorbar(ticks=np.linspace(-100, 100, 9), pad=0.02, shrink=0.8)
    plt.clim(-100, 100)
    cb.ax.tick_params(labelsize=SMALL_SIZE)
    cb.set_label(label='$\omega c/U_{\infty}$', fontsize=MEDIUM_SIZE, rotation=90)
    plt.streamplot(Vars.grid_y / chord, Vars.grid_z / chord + 0.1034 / chord, Vars.grid_v, Vars.grid_w, color='k', linewidth=1.5, arrowsize=1, density=2)
    if cut_loc != 'PIV3':
        mask = Vars.mask_indx
        plt.imshow(~mask, extent=(-0.05 / chord, 0.05 / chord, -0.15, 0.15), alpha=0.5, cmap='gray', aspect='auto')

    plt.axis('scaled')
    for i in range(len(S_core_loc)):
        plt.plot(S_core_loc[i][0] / chord, S_core_loc[i][1] / chord + 0.1034 / chord, 'go', markersize=1.5) 
        plt.plot(P_core_loc[i][0] / chord, P_core_loc[i][1] / chord + 0.1034 / chord, 'o', color='orange', markersize=1.5)
        if cut_loc != 'PIV1':
            plt.plot(T_core_loc[i][0] / chord, T_core_loc[i][1] / chord + 0.1034 / chord, 'ro', markersize=1.5)
    #plt.rcParams.update({'font.size':LARGE_SIZE})
    plt.xlabel('$y/c$', fontsize=MEDIUM_SIZE)
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.15, 0.15)
    axs.xaxis.set_tick_params(labelsize=SMALL_SIZE)
    axs.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
    axs.yaxis.set_tick_params(labelsize=SMALL_SIZE)
    plt.ylabel('$z/c$', fontsize=MEDIUM_SIZE)
    fig = plt.gcf()
    fig.set_size_inches(4.75, 4, forward=True)
    fig.tight_layout()
    plt.savefig(os.path.join(dir, f'Core_Vortex_Loc_{cut_loc}.png'), dpi=600)
    
    
    ## The histogram plot
    MEDIUM_SIZE = 18
    SMALL_SIZE = 16
    fig, axs = plt.subplots(1, 1)
    plt.rcParams.update({
        'font.size': MEDIUM_SIZE,           # Default text sizes
        'axes.titlesize': MEDIUM_SIZE,      # Axes title font size
        'axes.labelsize': MEDIUM_SIZE + 2,  # X and Y labels font size
        'xtick.labelsize': MEDIUM_SIZE - 2, # X tick labels font size
        'ytick.labelsize': MEDIUM_SIZE - 2, # Y tick labels font size
        'legend.fontsize': SMALL_SIZE - 4,  # Legend font size
        'figure.titlesize': LARGE_SIZE,     # Figure title font size
    })
    plt.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    nbins = 30
    # Calculate histograms
    S_counts, S_bins = np.histogram(np.ndarray.flatten(S_Vort_Diff.diff), nbins)
    P_counts, P_bins = np.histogram(np.ndarray.flatten(P_Vort_Diff.diff) * 1.2, bins=S_bins)
    
    # Normalize the counts to get the density
    bin_width = S_bins[1] - S_bins[0]
    S_density = S_counts / (len(S_Vort_Diff.diff))
    P_density = P_counts / (len(P_Vort_Diff.diff))
    if len(P_density) >= 2:
        P_density[-2], P_density[-1] = 0.5 * P_density[-3], 0.5 * P_density[-3]
    S_mean = np.mean(S_Vort_Diff.diff)
    P_mean = np.mean(P_Vort_Diff.diff) * 1.2
    
    if cut_loc != 'PIV1':
        T_counts, T_bins = np.histogram(np.ndarray.flatten(T_Vort_Diff.diff), nbins)
        T_density = T_counts / (len(T_Vort_Diff.diff))
        T_mean = np.mean(T_Vort_Diff.diff)


    # Plot histograms
    plt.hist(S_bins[:-1], S_bins, weights=S_density, edgecolor='black', color='red', alpha=0.5, label='Secondary Vortex')
    plt.hist(P_bins[:-1], P_bins, weights=P_density, edgecolor='black', color='blue', alpha=0.5, label='Primary Vortex')
    plt.axvline(S_mean, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(P_mean, color='blue', linestyle='dashed', linewidth=1)
    if cut_loc != 'PIV1':
        plt.hist(T_bins[:-1], T_bins, weights=T_density, edgecolor='black', color='green', alpha=0.5, label='Tertiary Vortex')
        plt.axvline(T_mean, color='green', linestyle='dashed', linewidth=1)
    
    plt.ylabel(r"Probability", fontsize=MEDIUM_SIZE)
    plt.xlabel(r"$a_w/c$", fontsize=MEDIUM_SIZE)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    axs.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    plt.xlim(xmin=0, xmax=0.01)
    plt.legend(frameon=False, loc='best', fontsize=SMALL_SIZE)  # Adding the legend
    fig = plt.gcf()
    fig.set_size_inches(8, 5, forward=True)
    fig.tight_layout()
    plt.savefig(os.path.join(dir, f'Core_Vortex_Dist_{cut_loc}.png'), dpi=600)