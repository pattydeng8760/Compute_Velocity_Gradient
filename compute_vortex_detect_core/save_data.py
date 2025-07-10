import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .utils import print


def save_data(vars, cut, P_core_loc, P_Vort_Diff, S_core_loc, S_Vort_Diff, T_core_loc, T_Vort_Diff, dir, tertiary=False):
    """
    Save vortex detection results to multiple file formats.
    
    This function saves the complete vortex detection analysis results including
    gridded flow fields, vortex core locations, and wandering statistics. Data
    is saved in both numpy binary format and MATLAB format for compatibility
    with different analysis tools.
    
    Args:
        vars (make_grid): Gridded flow field object containing interpolated data
        cut (str): Cut plane identifier for file naming
        P_core_loc (numpy.ndarray): Primary vortex core locations [N_timesteps, 2]
        P_Vort_Diff (vortex_trace): Primary vortex wandering statistics object
        S_core_loc (numpy.ndarray): Secondary vortex core locations [N_timesteps, 2]
        S_Vort_Diff (vortex_trace): Secondary vortex wandering statistics object
        T_core_loc (numpy.ndarray): Tertiary vortex core locations [N_timesteps, 2]
        T_Vort_Diff (vortex_trace): Tertiary vortex wandering statistics object
        dir (str): Output directory path for saving files
        tertiary (bool, optional): Whether tertiary vortex data exists. Defaults to False.
    
    Saves:
        Grid data (numpy format):
            - Grid_y_{cut}.npy: Y-coordinates normalized by chord length
            - Grid_z_{cut}.npy: Z-coordinates normalized and shifted by chord length
            - Grid_vort_{cut}.npy: Vorticity field on structured grid
            - Grid_u_{cut}.npy: U-velocity component on structured grid
            - Grid_v_{cut}.npy: V-velocity component on structured grid
            - Grid_w_{cut}.npy: W-velocity component on structured grid
            - Grid_mask_index_{cut}.npy: Airfoil surface mask (if applicable)
        
        Grid data (MATLAB format):
            - Grid_{cut}_Data.mat: Combined gridded data in MATLAB format
        
        Vortex core data (numpy format):
            - S_core_{cut}.npy: Secondary vortex core locations (normalized)
            - P_core_{cut}.npy: Primary vortex core locations (normalized)
            - S_core_{cut}_Diff.npy: Secondary vortex wandering amplitudes
            - P_core_{cut}_Diff.npy: Primary vortex wandering amplitudes
            - T_core_{cut}.npy: Tertiary vortex core locations (if tertiary=True)
            - T_core_{cut}_Diff.npy: Tertiary vortex wandering amplitudes (if tertiary=True)
    
    Note:
        - All spatial coordinates are normalized by chord length (0.3048 m)
        - Z-coordinates are shifted by 0.1034/chord for proper positioning
        - Airfoil mask is only saved for non-PIV3 cases
        - File naming follows the pattern: {type}_{cut}.{ext}
    """

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


