import numpy as np
import os
import glob
import multiprocessing
from multiprocessing import Pool, cpu_count
from antares import Reader
from window_bounds import get_window_boundaries
from .make_grid import make_grid
from .vortex_track import vortex, vortex_trace
from .save_data import save_data
from .utils import print_custom

def process_file_block(file_block, SV_WindowLL, SV_WindowUR, PV_WindowLL, PV_WindowUR, TV_WindowLL, TV_WindowUR, cut_loc, block_num):
    """
    Process a block of HDF5 files in parallel for vortex detection.
    
    This function is designed for parallel execution and processes a subset of timestep files
    to detect vortex cores. It extracts flow field data from each file, applies vortex detection
    algorithms, and aggregates the results for the entire block.
    
    Args:
        file_block (list): List of HDF5 file paths to process in this block
        SV_WindowLL (list): Lower-left corner [y, z] of secondary vortex detection window
        SV_WindowUR (list): Upper-right corner [y, z] of secondary vortex detection window
        PV_WindowLL (list): Lower-left corner [y, z] of primary vortex detection window
        PV_WindowUR (list): Upper-right corner [y, z] of primary vortex detection window
        TV_WindowLL (list): Lower-left corner [y, z] of tertiary vortex detection window
        TV_WindowUR (list): Upper-right corner [y, z] of tertiary vortex detection window
        cut_loc (str): Cut plane identifier (e.g., 'PIV1', '030_TE')
        block_num (int): Block number for progress tracking and identification
    
    Returns:
        dict: Dictionary containing aggregated results with keys:
            - 'u', 'v', 'w': Velocity components for all timesteps (numpy.ndarray)
            - 'vort_x': Vorticity field for all timesteps (numpy.ndarray)
            - 'S_core_loc': Secondary vortex core locations (numpy.ndarray)
            - 'P_core_loc': Primary vortex core locations (numpy.ndarray)
            - 'T_core_loc': Tertiary vortex core locations (numpy.ndarray)
            - 'y', 'z': Spatial coordinates from the first file (numpy.ndarray)
            - 'block_num': Block identifier for debugging (int)
    
    Note:
        Progress is reported every 10 files processed within the block.
        Vorticity is scaled by the factor (0.305/30) for normalization.
        Tertiary vortex detection is conditional based on the cut location.
    """
    aggregated_u, aggregated_v, aggregated_w, aggregated_vort = [], [], [], []
    S_core_loc_block, P_core_loc_block, T_core_loc_block = [], [], []
    
    # Variables to extract 'y' and 'z' as positional coordinates
    y, z = None, None
    
    # Progress tracking
    total_files_in_block = len(file_block)
    
    # Print files being assessed for this block
    print_custom(f"    Block {block_num}: Processing {total_files_in_block} files")
    for file_idx, file_path in enumerate(file_block):
        file_name = os.path.basename(file_path)
        print_custom(f"        File {file_idx + 1}/{total_files_in_block}: {file_name}")
        
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
                    print_custom(f"Block {block_num}: 'y' and 'z' differ in file {file_path}. Using the first file's 'y' and 'z'.")
            
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
            print_custom(f"Block {block_num}: Error processing file {file_path}: {e}")
            continue  # Skip to the next file
    
    # Final progress report for the block
    print_custom(f"    Block {block_num}: Completed processing {total_files_in_block} files")
    
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
        'z': z,
        'block_num': block_num
    }

def detect_vortex(source_dir, cut, alpha, method='area', nb_tasks=None, max_file=None, output_dir='./'):
    """
    Detect vortices from CFD/PIV data using parallel processing and multiple detection algorithms.
    
    This is the main vortex detection function that orchestrates the complete workflow:
    1. Loads window boundaries for the specified cut and angle of attack
    2. Discovers and sorts all HDF5 timestep files
    3. Distributes files across parallel workers for processing
    4. Aggregates results from all workers
    5. Computes time-averaged flow fields and vortex statistics
    6. Saves processed data to output files
    
    The function supports different vortex detection methods and can handle both
    LES simulation data and PIV experimental data with appropriate scaling.
    
    Args:
        source_dir (str): Directory path containing HDF5 timestep files (*.h5)
        cut (str): Cut plane identifier (e.g., 'PIV1', 'PIV2', '030_TE', '040_TE')
        alpha (int): Angle of attack in degrees for window boundary selection
        method (str, optional): Vortex detection algorithm. Defaults to 'area'.
            - 'max': Find vorticity maximum within detection windows
            - 'precise': Use structured grid and connected component analysis
            - 'area': Find geometric center of largest vortical region
        nb_tasks (int, optional): Number of parallel processes. Defaults to CPU count.
        max_file (int, optional): Limit number of files processed (for testing). Defaults to None.
        output_dir (str, optional): Directory for saving results. Defaults to './'.
    
    Returns:
        tuple: Seven-element tuple containing:
            - S_core_loc (numpy.ndarray): Secondary vortex core locations [N_timesteps, 2]
            - S_Vort_Diff (vortex_trace): Secondary vortex wandering statistics
            - P_core_loc (numpy.ndarray): Primary vortex core locations [N_timesteps, 2]
            - P_Vort_Diff (vortex_trace): Primary vortex wandering statistics  
            - T_core_loc (numpy.ndarray): Tertiary vortex core locations [N_timesteps, 2]
            - T_Vort_Diff (vortex_trace): Tertiary vortex wandering statistics
            - Vars (make_grid): Time-averaged flow field on structured grid
    
    Raises:
        ValueError: If window boundaries cannot be found for the specified cut/alpha
        FileNotFoundError: If no HDF5 files are found in the source directory
        
    Example:
        >>> results = detect_vortex('/path/to/data', 'PIV1', 10, method='precise', nb_tasks=8)
        >>> S_loc, S_diff, P_loc, P_diff, T_loc, T_diff, grid = results
        >>> print_custom(f"Detected {len(S_loc)} timesteps of secondary vortex data")
    
    Note:
        - Window boundaries are loaded from the window_bounds module
        - Vorticity is normalized by (0.305/30) factor during processing
        - Tertiary vortex detection depends on the cut location
        - Results are automatically saved to numpy and MATLAB files
    """
    print_custom('\n----> Data Source Information:')
    print_custom(f'    Source directory: {source_dir}')
    print_custom(f'    Cut location: {cut}')
    
    # Get window boundaries for the given cut location
    try:
        window_boundaries = get_window_boundaries(cut, str(alpha))
    except ValueError as e:
        print_custom(str(e))
        raise
    
    SV_WindowLL = window_boundaries['SV_WindowLL']
    SV_WindowUR = window_boundaries['SV_WindowUR']
    PV_WindowLL = window_boundaries['PV_WindowLL']
    PV_WindowUR = window_boundaries['PV_WindowUR']
    TV_WindowLL = window_boundaries['TV_WindowLL']
    TV_WindowUR = window_boundaries['TV_WindowUR']

    # Glob all .h5 files in the source directory
    source_files = sorted(glob.glob(os.path.join(source_dir, '*.h5')))
    if max_file is not None:
        source_files = source_files[:max_file]
    
    total_files = len(source_files)
    print_custom(f'    Total number of files to process: {total_files}')

    if total_files == 0:
        print_custom("No .h5 files found in the source directory.")
        raise FileNotFoundError("No .h5 files found in the source directory.")

    # Determine the number of parallel tasks
    avail = cpu_count()
    if nb_tasks is None:
        nb_tasks = avail
    nproc = min(avail, nb_tasks)
    if nproc > total_files:
        nproc = total_files
    
    print_custom('\n----> Parallel Processing Configuration:')
    print_custom(f'    Number of available parallel compute processes: {nproc}')
    print_custom(f'    Number of parallel tasks (blocks): {nproc}')

    # Split the data_files into nproc roughly equal blocks
    file_blocks = np.array_split(source_files, nproc)
    print_custom(f'    Data files split into {nproc} blocks.')
    
    # Create blocks with additional parameters for starmap
    blocks = [(file_block, SV_WindowLL, SV_WindowUR, PV_WindowLL, PV_WindowUR, TV_WindowLL, TV_WindowUR, cut, block_num) 
              for block_num, file_block in enumerate(file_blocks)]

    print_custom('\n----> Performing Parallel Vortex Detection...')
    
    # Process blocks in parallel
    with Pool(nproc) as pool:
        results = pool.starmap(process_file_block, blocks)
    
    print_custom('\n----> File Processing Results:')
    print_custom('    File processing complete.')
    
    # Initialize lists to collect results
    aggregated_u, aggregated_v, aggregated_w, aggregated_vort = [], [], [], []
    S_core_loc, P_core_loc, T_core_loc = [], [], []
    y_master, z_master = None, None
    
    # Stitch the results together
    for result in results:
        if result['y'] is not None and result['z'] is not None:
            if y_master is None and z_master is None:
                y_master = result['y']
                z_master = result['z']
            else:
                if not (np.array_equal(y_master, result['y']) and np.array_equal(z_master, result['z'])):
                    print_custom("Inconsistent 'y' and 'z' across files. Using the first file's 'y' and 'z'.")
        
        aggregated_u.append(result['u'])
        aggregated_v.append(result['v'])
        aggregated_w.append(result['w'])
        aggregated_vort.append(result['vort_x'])
        S_core_loc.extend(result['S_core_loc'])
        P_core_loc.extend(result['P_core_loc'])
        
        # Check for tertiary vortex
        if TV_WindowLL is not None and TV_WindowUR is not None and len(TV_WindowLL) == 2 and len(TV_WindowUR) == 2:
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
    if y_master is None or z_master is None:
        print_custom("No 'y' and 'z' data collected from any files.")
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
    S_Vortex = vortex('Secondary', SV_WindowLL, SV_WindowUR, y_master, z_master, mean_u, mean_vort, method, -20)
    P_Vortex = vortex('Primary', PV_WindowLL, PV_WindowUR, y_master, z_master, mean_u, mean_vort, method, -20)
    S_Vort_Diff = vortex_trace(S_Vortex.core.core_loc, S_core_loc)
    P_Vort_Diff = vortex_trace(P_Vortex.core.core_loc, P_core_loc)
    
    if tertiary:
        T_Vortex = vortex('Tertiary', TV_WindowLL, TV_WindowUR, y_master, z_master, mean_u, mean_vort, method, -20)
        T_Vort_Diff = vortex_trace(T_Vortex.core.core_loc, T_core_loc)
    else:
        T_Vortex, T_Vort_Diff = [], [] 
    
    # Saving data
    print_custom('\n----> Saving Results:')
    print_custom(f'    Saving processed data to: {output_dir}')
    save_data(Vars, cut, P_core_loc, P_Vort_Diff, S_core_loc, S_Vort_Diff, T_core_loc, T_Vort_Diff, output_dir, tertiary)
    print_custom('    Data saved successfully')
    return S_core_loc, S_Vort_Diff, P_core_loc, P_Vort_Diff, T_core_loc, T_Vort_Diff, Vars