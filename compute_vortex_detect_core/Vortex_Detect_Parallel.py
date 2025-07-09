####################################################################################
# Applying Vortex Detection to transient data
# The subfile with all required functions
# Author: Patrick Deng
# Refactored to use multiprocessing.Pool with data blocks and proper parameter passing
####################################################################################
# The required modules
import os
import numpy as np
import sys
from antares import *
import h5py
import copy
import scipy
import glob
import builtins
import shutil
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import griddata
from scipy.ndimage.measurements import label
from itertools import combinations
import multiprocessing
from functools import partial
import time
from sys import argv
import logging
import logging.handlers
from multiprocessing import Queue
from window_bounds import get_window_boundaries

# Initialize a multiprocessing-safe logging queue
log_queue = Queue()

# Configure the root logger to use the logging queue
queue_handler = logging.handlers.QueueHandler(log_queue)
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set to DEBUG to capture detailed logs
logger.addHandler(queue_handler)

# Define a handler for the listener to write log messages to a file
file_handler = logging.FileHandler('logging_parallel_extract.txt')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
file_handler.setFormatter(formatter)

# Optionally, add a StreamHandler to output logs to the console
# Uncomment the following lines if you want console logs
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# listener = logging.handlers.QueueListener(log_queue, file_handler, stream_handler)

listener = logging.handlers.QueueListener(log_queue, file_handler)
listener.start()

# Define the vortex-related classes
class vortex:
    def __init__(self, pos: str, A: list, B: list, y, z, u, vort, choice: str, level:float = -30) -> None:
        assert choice in ['max', 'precise', 'area'], "Invalid choice for vortex detection."
        self.position = pos
        self.location = self.vortex_loc(A, B)
        self.core = self.vortex_core(y, z, u, vort, self.location.y, self.location.z, choice, level)
        
    class vortex_loc:
        def __init__(self, A: list, B: list):
            self.y = [np.min([A[0], B[0]]), np.max([A[0], B[0]])]
            self.z = [np.min([A[1], B[1]]), np.max([A[1], B[1]])]
            
    class vortex_core:
        def __init__(self, y, z, u, vort, y_lim: list, z_lim: list, choice: str, level: int):
            if choice == 'max':
                mask = (y >= np.min(y_lim)) & (y <= np.max(y_lim)) & (z >= np.min(z_lim)) & (z <= np.max(z_lim))
                dummy = vort * mask
                if np.all(dummy == 0):
                    logger.warning("All vortex values masked in 'max' choice.")
                    self.core_loc = [[np.nan, np.nan]]
                    self.core_mag = [np.nan]
                else:
                    min_idx = np.argmin(dummy)
                    self.core_loc = [[y[min_idx], z[min_idx]]]
                    self.core_mag = [vort[min_idx]]
            elif choice == 'precise':
                n = 200
                vars = make_grid(n, y_lim, z_lim, y, z, u, [], [], vort, False)
                index_airf = np.where(np.abs(vars.grid_u) < 1e-3)
                vars.grid_vort[index_airf],vars.grid_u[index_airf]  = float(0),float(0)
                mask0 =  (vars.grid_u >= 1)
                mask1 = (vars.grid_vort <= level) if level <= -15 else (vars.grid_vort >= level) 
                mask2 = mask0 & mask1
                bounds, size_bounds, largest_area = find_squares(mask2)
                if not largest_area:
                    logger.warning("No largest area found in 'precise' choice.")
                    self.core_loc = [[np.nan, np.nan]]
                    self.core_mag = [np.nan]
                else:
                    mask3 = np.zeros_like(mask1)
                    mask3[largest_area[0][0]:largest_area[1][0], largest_area[0][1]:largest_area[1][1]] = 1
                    mask = mask3 & mask2
                    dummy = vars.grid_vort * mask
                    index = np.argwhere(dummy == np.min(dummy))
                    if index.size == 0:
                        logger.warning("No minimum found in 'precise' choice for vortex core.")
                        self.core_loc = [[np.nan, np.nan]]
                        self.core_mag = [np.nan]
                    else:
                        y_core = vars.grid_y[index[0,0], index[0,1]]
                        z_core = vars.grid_z[index[0,0], index[0,1]]
                        self.core_loc = [[y_core, z_core]]
                        self.core_mag = [vort[np.argmin(dummy)]]
            elif choice == 'area':
                n = 200
                vars = make_grid(n, y_lim, z_lim, y, z, u, [], [], vort, False)
                index_airf = np.where(np.abs(vars.grid_u) < 1e-3)
                vars.grid_vort[index_airf],vars.grid_u[index_airf]  = float(0),float(0)
                mask0 =  (vars.grid_u >= 1)
                mask1 = (vars.grid_vort <= level) if level <= -15 else (vars.grid_vort >= level) 
                mask2 = mask0 & mask1
                bounds, size_bounds, largest_area = find_squares(mask2)
                if not largest_area:
                    logger.warning("No largest area found in 'area' choice.")
                    self.core_loc = [[np.nan, np.nan]]
                    self.core_mag = [np.nan]
                else:
                    mask3 = np.zeros_like(mask1)
                    mask3[largest_area[0][0]:largest_area[1][0], largest_area[0][1]:largest_area[1][1]] = 1
                    mask = mask2 & mask3
                    dummy = vars.grid_vort * mask
                    index = [(largest_area[0][0] + largest_area[1][0]) / 2, (largest_area[0][1] + largest_area[1][1]) / 2]
                    y_core = vars.grid_y[int(index[0]), int(index[1])]
                    z_core = vars.grid_z[int(index[0]), int(index[1])]
                    self.core_loc = [[y_core, z_core]]
                    self.core_mag = [vort[np.argmin(dummy)]]

class vortex_trace:
    def __init__(self, mean, inst):
        self.calculate_trace(mean, inst)
        
    def calculate_trace(self, mean, inst):
        diff = np.zeros((np.shape(inst)[0], 1))
        for i in range(np.shape(inst)[0]):
            diff[i] = np.sqrt((mean[0][0] - inst[i][0])**2 + (mean[0][1] - inst[i][1])**2) 
        self.diff = diff

# Function to find connected squares
def find_squares(a):
    # Find ones
    ones = [(i, j) for i, row in enumerate(a) for j, val in enumerate(row) if val]
    if not ones:
        logger.warning("No connected squares found.")
        return [], [], []
    # Make graph of connected ones
    graph = {a: [] for a in ones}
    for a, b in combinations(ones, 2):
        if abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1:
            graph[a].append(b)
            graph[b].append(a)
    # Find connected components in graph
    components = []
    for a, a_neigh in graph.items():
        if any(a in c for c in components):
            continue
        component = set()
        component.add(a)
        pending = a_neigh.copy()
        while pending:
            b = pending.pop()
            if b not in component:
                component.add(b)
                pending.extend(graph[b])
        components.append(component)
    # Find bounds for each component
    bounds = [((min(a[0] for a in c), min(a[1] for a in c)),
               (max(a[0] for a in c), max(a[1] for a in c)))
              for c in components]
    # The size of the boundary
    size_bounds = [(bounds[i][1][1] - bounds[i][0][1]) * (bounds[i][1][0] - bounds[i][0][0]) for i in range(len(bounds))]
    if not size_bounds:
        logger.warning("No size bounds calculated.")
        largest_area = []
    else:
        # The index of the region with the largest area
        largest_area = bounds[np.argmax(size_bounds)]
    return bounds, size_bounds, largest_area

class make_grid:
    def __init__(self, n: int, y_bnd: list, z_bnd: list, y, z, u, v, w, vort, airfoil: bool):
        self.calculate_grid(n, y_bnd, z_bnd, y, z, u, v, w, vort, airfoil)
        
    def calculate_grid(self, n: int, y_bnd: list, z_bnd: list, y, z, u, v, w, vort, airfoil: bool):
        # Interpolating the data from unstructured grid to rectangular for plotting
        y_lin = np.linspace(min(y_bnd), max(y_bnd), num=n)
        z_lin = np.linspace(min(z_bnd), max(z_bnd), num=n)
        self.grid_y, self.grid_z = np.meshgrid(y_lin, z_lin)
        self.y, self.z = y, z
        self.u,self.v,self.w = u,v,w
        self.vort = vort
        # Interpolating
        self.grid_u = griddata(np.transpose([self.y, self.z]), self.u, (self.grid_y, self.grid_z), method='linear')
        self.grid_v = [] if len(v) == 0 else griddata(np.transpose([self.y, self.z]), self.v, (self.grid_y, self.grid_z), method='linear')
        self.grid_w = [] if len(w) == 0 else griddata(np.transpose([self.y, self.z]), self.w, (self.grid_y, self.grid_z), method='linear')
        self.grid_vort = [] if len(vort) == 0 else griddata(np.transpose([self.y, self.z]), self.vort, (self.grid_y, self.grid_z), method='linear')
        if airfoil:
            # Masking the values inside the airfoil surface as zero
            index_airf = np.where(np.abs(self.grid_u) < 1e-3)
            self.grid_u[index_airf] = float(0)
            if isinstance(self.grid_v, np.ndarray):
                self.grid_v[index_airf] = float(0)
            self.grid_w[index_airf] = float(0)
            self.grid_vort[index_airf] = float('nan')
            self.mask_y = [min(self.grid_y[index_airf]), max(self.grid_y[index_airf])]
            self.mask_z = [min(self.grid_z[index_airf]), max(self.grid_z[index_airf])]
            self.mask_indx = np.flip(np.isnan(self.grid_vort), axis=0)

# Worker function to process a block of files
def process_block(block, SV_WindowLL, SV_WindowUR, PV_WindowLL, PV_WindowUR, TV_WindowLL, TV_WindowUR, cut_loc):
    """
    Processes a block of files and returns aggregated results.

    Parameters:
        block (list): List of file paths to process.
        SV_WindowLL, SV_WindowUR, PV_WindowLL, PV_WindowUR, TV_WindowLL, TV_WindowUR (list): Window boundaries.
        cut_loc (str): Cut location identifier.

    Returns:
        dict: Aggregated data from the block.
    """
    aggregated_u = []
    aggregated_v = []
    aggregated_w = []
    aggregated_vort = []
    S_core_loc_block = []
    P_core_loc_block = []
    T_core_loc_block = []
    
    # Variables to extract 'y' and 'z'; assuming consistency, extract from the first file
    y = None
    z = None
    
    for file_path in block:
        try:
            r = Reader('hdf_antares')
            r['filename'] = file_path
            base = r.read()
            
            # Extract data
            y_file = base['0000']['0000']['y']
            z_file = base['0000']['0000']['z']
            vort_x = base['0000']['0000']['vort_x'] * 0.305 / 30
            u = base['0000']['0000']['u']
            v_val = base['0000']['0000']['v']
            w = base['0000']['0000']['w']
            
            if y is None and z is None:
                y = y_file
                z = z_file
            else:
                # Optionally, verify that 'y' and 'z' are consistent
                if not (np.array_equal(y, y_file) and np.array_equal(z, z_file)):
                    logger.warning(f"'y' and 'z' differ in file {file_path}. Using the first file's 'y' and 'z'.")
            
            # Initialize vortex objects
            S_Vortex = vortex('Secondary', SV_WindowLL, SV_WindowUR, y, z, u, vort_x, 'area', -18)
            P_Vortex = vortex('Primary', PV_WindowLL, PV_WindowUR, y, z, u, vort_x, 'area', -18)
            if cut_loc != 'PIV1' and cut_loc != '030_TE' and cut_loc != '040_TE' and cut_loc != '050_TE' and cut_loc != '060_TE' and cut_loc != '070_TE':
                flag_tertiary = True
            else:
                flag_tertiary = False
            T_Vortex = vortex('Tertiary', TV_WindowLL, TV_WindowUR, y, z, u, vort_x, 'precise', -20) if flag_tertiary else None
            
            # Append data
            aggregated_u.append(u)
            aggregated_v.append(v_val)
            aggregated_w.append(w)
            aggregated_vort.append(vort_x)
            
            # Store core locations
            S_core_loc_block.append(S_Vortex.core.core_loc[0])
            P_core_loc_block.append(P_Vortex.core.core_loc[0])
            if T_Vortex:
                T_core_loc_block.append(T_Vortex.core.core_loc[0])
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue  # Skip to the next file
    
    # Convert lists to numpy arrays
    aggregated_u = np.array(aggregated_u)
    aggregated_v = np.array(aggregated_v)
    aggregated_w = np.array(aggregated_w)
    aggregated_vort = np.array(aggregated_vort)
    S_core_loc_block = np.array(S_core_loc_block)
    P_core_loc_block = np.array(P_core_loc_block)
    T_core_loc_block = np.array(T_core_loc_block) if cut_loc != 'PIV1' else np.array([])
    
    # Return aggregated data as a dictionary
    return {
        'u': aggregated_u,
        'v': aggregated_v,
        'w': aggregated_w,
        'vort_x': aggregated_vort,
        'S_core_loc': S_core_loc_block,
        'P_core_loc': P_core_loc_block,
        'T_core_loc': T_core_loc_block,
        'y': y,
        'z': z
    }

def Detect_Vortex(source_dir, cut, alpha, method='area', nb_tasks=None, max_file:int=None, output_dir:str='./'):
    """
    Detects vortices in the given source directory using parallel processing.

    Parameters:
        source_dir (str): Path to the directory containing .h5 files.
        cut (str): Cut location identifier.
        method (str): Method for vortex detection ('max', 'precise', 'area', etc.).
        nb_tasks (int, optional): Number of parallel tasks (blocks). Defaults to the number of CPU cores.

    Returns:
        tuple: Aggregated vortex core locations and other related data.
    """
    logger.info(f'The source directory is: {source_dir}')
    logger.info(f'The cut location is: {cut}')
    
    # Get window boundaries for the given cut location
    try:
        window_boundaries = get_window_boundaries(cut, str(alpha))
    except ValueError as e:
        logger.error(str(e))
        raise
    
    SV_WindowLL = window_boundaries['SV_WindowLL']
    SV_WindowUR = window_boundaries['SV_WindowUR']
    PV_WindowLL = window_boundaries['PV_WindowLL']
    PV_WindowUR = window_boundaries['PV_WindowUR']
    TV_WindowLL = window_boundaries['TV_WindowLL']
    TV_WindowUR = window_boundaries['TV_WindowUR']

    # Glob all .h5 files in the source directory
    source_files = sorted(glob.glob(os.path.join(source_dir, '*.h5')))[0:max_file] if max_file is not None else sorted(glob.glob(os.path.join(source_dir, '*.h5')))
    total_files = len(source_files)
    logger.info(f"Total number of files to process: {total_files}")

    if total_files == 0:
        logger.error("No .h5 files found in the source directory.")
        raise FileNotFoundError("No .h5 files found in the source directory.")

    # Determine the number of parallel tasks
    if nb_tasks is None:
        nb_tasks = multiprocessing.cpu_count()
    if nb_tasks > total_files:
        nb_tasks = total_files
    logger.info(f"Number of parallel tasks (blocks): {nb_tasks}")

    # Split the data_files into nb_tasks roughly equal blocks
    blocks = np.array_split(source_files, nb_tasks)
    logger.info(f"Data files split into {nb_tasks} blocks.")

    # Initialize lists to collect results
    aggregated_u, aggregated_v, aggregated_w, aggregated_vort = [], [], [], []
    S_core_loc, P_core_loc, T_core_loc = [], [], []
    
    # Create a multiprocessing Pool
    pool = multiprocessing.Pool(processes=nb_tasks)
    
    try:
        # Use functools.partial to pass window parameters and cut_loc to process_block
        process_func = partial(
            process_block, 
            SV_WindowLL=SV_WindowLL, 
            SV_WindowUR=SV_WindowUR, 
            PV_WindowLL=PV_WindowLL, 
            PV_WindowUR=PV_WindowUR, 
            TV_WindowLL=TV_WindowLL, 
            TV_WindowUR=TV_WindowUR, 
            cut_loc=cut
        )
        
        # Map the process_block function to each block
        results = pool.map(process_func, blocks)
    finally:
        pool.close()
        pool.join()
        logger.info("File interpolation complete.")
    
    # Stitch the results together
    for result in results:
        if result['y'] is not None and result['z'] is not None:
            if 'y_master' not in locals():
                y_master = result['y']
                z_master = result['z']
            else:
                if not (np.array_equal(y_master, result['y']) and np.array_equal(z_master, result['z'])):
                    logger.warning("Inconsistent 'y' and 'z' across files. Using the first file's 'y' and 'z'.")
        
        aggregated_u.append(result['u'])
        aggregated_v.append(result['v'])
        aggregated_w.append(result['w'])
        aggregated_vort.append(result['vort_x'])
        S_core_loc.extend(result['S_core_loc'])
        P_core_loc.extend(result['P_core_loc'])
        if TV_WindowLL is not None and TV_WindowUR is not None and len(TV_WindowLL) ==2 and len(TV_WindowUR) ==2:
            tertiary = True
        else: 
            tertiary = False
        if tertiary:
            T_core_loc.extend(result['T_core_loc'])
            
    # Convert aggregated lists to numpy arrays
    aggregated_u = np.concatenate(aggregated_u, axis=0)
    aggregated_v = np.concatenate(aggregated_v, axis=0)
    aggregated_w = np.concatenate(aggregated_w, axis=0)
    aggregated_vort = np.concatenate(aggregated_vort, axis=0)
    S_core_loc = np.array(S_core_loc)
    P_core_loc = np.array(P_core_loc)
    T_core_loc = np.array(T_core_loc) if tertiary else np.array([])

    # Check if 'y_master' and 'z_master' are defined
    if 'y_master' not in locals() or 'z_master' not in locals():
        logger.error("No 'y' and 'z' data collected from any files.")
        raise ValueError("No 'y' and 'z' data collected from any files.")
    
    # The average vortex core locations
    mean_vort = np.mean(aggregated_vort, axis=0)
    mean_u = np.mean(aggregated_u, axis=0)
    mean_v = np.mean(aggregated_v, axis=0)
    mean_w = np.mean(aggregated_w, axis=0)
    
    n = 500
    y_bnd = [-0.05, 0.05]
    z_bnd = [-0.16, -0.06]
    airfoil = True if cut != 'PIV3' else False
    Vars = make_grid(n, y_bnd, z_bnd, y_master, z_master, mean_u, mean_v, mean_w, mean_vort, airfoil)
    
    # Initialize vortex objects with mean values
    S_Vortex = vortex('Secondary', SV_WindowLL, SV_WindowUR, y_master, z_master, mean_u, mean_vort, method,-20)
    P_Vortex = vortex('Primary', PV_WindowLL, PV_WindowUR, y_master, z_master, mean_u, mean_vort, method,-20)
    S_Vort_Diff = vortex_trace(S_Vortex.core.core_loc, S_core_loc)
    P_Vort_Diff = vortex_trace(P_Vortex.core.core_loc, P_core_loc)
    
    if tertiary:
        T_Vortex = vortex('Tertiary', TV_WindowLL, TV_WindowUR, y_master, z_master, mean_u, mean_vort, method,-20)
        T_Vort_Diff = vortex_trace(T_Vortex.core.core_loc, T_core_loc)
    else:
        T_Vortex, T_Vort_Diff = [], [] 
    
    # Saving data
    save_data(Vars, cut, P_core_loc, P_Vort_Diff, S_core_loc, S_Vort_Diff, T_core_loc, T_Vort_Diff, output_dir,tertiary)
    return S_core_loc, S_Vort_Diff, P_core_loc, P_Vort_Diff, T_core_loc, T_Vort_Diff, Vars

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


def main():
    try:
        # The directory information
        t = time.time()
        cut_loc = argv[1]
        #base_dirs = {
        #     #'PIV1': '/project/p/plavoie/denggua1/BBDB_10AOA/RUN_ZONE/Isosurface/',
        #     'PIV1': '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface/',
        #     #'PIV2': '/project/p/plavoie/denggua1/BBDB_10AOA/RUN_ZONE/Isosurface/',
        #     'PIV2': '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface/',
        #     #'PIV3': '/project/p/plavoie/denggua1/BBDB_10AOA/Isosurface/'
        #     'PIV3': '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface/',
        # }
        base_dirs = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface/'
        # Construct the source directory
        source_dir = os.path.join(base_dirs, f'Cut_{cut_loc}_VGT')
        
        #source_dir = '/project/p/plavoie/denggua1/BBDB_10AOA/RUN_ZONE/Isosurface/Cut_PIV1_VGT'
        #source_dir = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface/Cut_PIV3_VGT/'
        #folder_name = os.path.basename(os.path.normpath(source_dir))
        #cut_loc = folder_name.split('_')[1]
        #cut_loc = source_dir.split('/')[-1].split('_')[1]
        #cut_loc = os.path.basename(os.path.dirname(source_dir))
        sol_dir = f'Vortex_Detect_Results_{cut_loc}'
        os.makedirs(sol_dir, exist_ok=True)
        chord = 0.3048
        
        # Number of parallel tasks (blocks)
        nb_tasks = multiprocessing.cpu_count()  # Automatically set to the number of CPU cores
        logger.info(f"Using {nb_tasks} parallel tasks.")
        alpha  = 5
        # Detect vortices
        S_core_loc, S_Vort_Diff, P_core_loc, P_Vort_Diff, T_core_loc, T_Vort_Diff, Vars = Detect_Vortex(source_dir, cut_loc, alpha,'precise', nb_tasks=None, max_file=None, output_dir=sol_dir)
        
        # Plot results
        Plot_Result(S_core_loc, S_Vort_Diff, P_core_loc, P_Vort_Diff, T_core_loc, T_Vort_Diff, Vars, cut_loc, sol_dir, chord=chord)
        
        elapsed = time.time() - t
        logger.info(f'The total calculation time is: {elapsed:1.0f} s')
    
    finally:
        # Ensure that the listener is stopped even if an error occurs
        listener.stop()

if __name__ == "__main__":
    main()
