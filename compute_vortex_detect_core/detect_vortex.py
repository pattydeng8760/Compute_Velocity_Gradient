import numpy as np
import os
import glob

def detect_vortex(source_dir, cut, alpha, method='area', nb_tasks=None, max_file:int=None, output_dir:str='./'):
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
    S_Vortex = vortex('Secondary', SV_WindowLL, SV_WindowUR, y_master, z_master, mean_u, mean_vort, method, -20)
    P_Vortex = vortex('Primary', PV_WindowLL, PV_WindowUR, y_master, z_master, mean_u, mean_vort, method, -20)
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
    aggregated_u, aggregated_v, aggregated_w, aggregated_vort = [], [], [], []
    S_core_loc_block, P_core_loc_block, T_core_loc_block = [], [], []
    
    # Variables to extract 'y' and 'z' as positional coordinates
    y,z = None, None
    
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