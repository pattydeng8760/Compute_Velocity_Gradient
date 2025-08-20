import numpy as np
import os
import glob
import multiprocessing
from multiprocessing import Pool, cpu_count
from antares import Reader
from window_bounds import get_window_boundaries_PIV
from .vortex_track import vortex, vortex_trace
from .save_data import save_data
from .utils import print

def process_file_block_piv(file_block, SV_WindowLL, SV_WindowUR, PV_WindowLL, PV_WindowUR, TV_WindowLL, TV_WindowUR, cut_loc, var, level, method, block_num):
    """
    Process a block of PIV HDF5 files in parallel for vortex detection.
    
    This function is designed for parallel execution and processes a subset of timestep files
    to detect vortex cores from PIV data. Unlike LES data, PIV data is already structured
    so no re-gridding is required. The coordinate system is different from LES:
    - PIV x-direction = wall-normal (LES y-direction)
    - PIV y-direction = span (LES z-direction)
    - PIV z-direction = streamwise (LES x-direction)
    
    Args:
        file_block (list): List of HDF5 file paths to process in this block
        SV_WindowLL (list): Lower-left corner [x, y] of secondary vortex detection window (PIV coords)
        SV_WindowUR (list): Upper-right corner [x, y] of secondary vortex detection window (PIV coords)
        PV_WindowLL (list): Lower-left corner [x, y] of primary vortex detection window (PIV coords)
        PV_WindowUR (list): Upper-right corner [x, y] of primary vortex detection window (PIV coords)
        TV_WindowLL (list): Lower-left corner [x, y] of tertiary vortex detection window (PIV coords)
        TV_WindowUR (list): Upper-right corner [x, y] of tertiary vortex detection window (PIV coords)
        cut_loc (str): Cut plane identifier (e.g., 'PIV1', 'PIV2', 'PIV3')
        var (str): Variable name for vortex core tracking (vort_x or lambda2)
        level (float): threshold for vortex detection (e.g., -30)
        method (str): Vortex detection method ('max', 'precise', 'area')
        block_num (int): Block number for progress tracking and identification
    
    Returns:
        dict: Dictionary containing aggregated results with keys:
            - 'u', 'v', 'w': Velocity components for all timesteps (numpy.ndarray)
            - 'vort_x': Vorticity field for all timesteps (numpy.ndarray)
            - 'S_core_loc': Secondary vortex core locations (numpy.ndarray)
            - 'P_core_loc': Primary vortex core locations (numpy.ndarray)
            - 'T_core_loc': Tertiary vortex core locations (numpy.ndarray)
            - 'x', 'y': Spatial coordinates from the first file (numpy.ndarray) - PIV coordinates
            - 'block_num': Block identifier for debugging (int)
    
    Note:
        Progress is reported every 10 files processed within the block.
        Vorticity is scaled by the factor (0.305/30) for normalization.
        For PIV data, we work directly with the structured grid without interpolation.
    """
    aggregated_u, aggregated_v, aggregated_w, aggregated_vort = [], [], [], []
    S_core_loc_block, P_core_loc_block, T_core_loc_block = [], [], []
    
    # Variables to extract PIV coordinates (x=wall-normal, y=span)
    x, y = None, None
    
    # Initialize flag_tertiary
    flag_tertiary = False
    
    # Progress tracking
    total_files_in_block = len(file_block)
    
    # Print files being assessed for this block
    print(f"    Block {block_num}: Processing {total_files_in_block} PIV files")
    for file_idx, file_path in enumerate(file_block):
        file_name = os.path.basename(file_path)
        print(f"        File {file_idx + 1}/{total_files_in_block}: {file_name}")
        
        try:
            
            r = Reader('hdf_antares')
            r['filename'] = file_path
            base = r.read()
            
            # Extract PIV data - note coordinate system difference
            x_file = base['0000']['0000']['x']  # Wall-normal direction (LES y)
            y_file = base['0000']['0000']['y']  # Span direction (LES z)
            vort_x = base['0000']['0000']['vort_x']*0.3048/30  # Streamwise vorticity
            u = base['0000']['0000']['u']  # Wall-normal velocity
            v = base['0000']['0000']['v']  # Span velocity
            w = base['0000']['0000']['w']  # Streamwise velocity
            l2 = base['0000']['0000']['lambda2']*(0.3048**2)/(30**2)  # lambda 2 criterion
            Q = base['0000']['0000']['Q']*(0.3048**2)/(30**2) # Q criterion
            if x is None and y is None:
                x = x_file
                y = y_file
            else:
                # Optionally, verify that coordinates are consistent
                if not (np.array_equal(x, x_file) and np.array_equal(y, y_file)):
                    print(f"Block {block_num}: PIV coordinates differ in file {file_path}. Using the first file's coordinates.")
            
            if var == 'vort_x':
                var_data = vort_x
            elif var == 'lambda2':
                var_data = l2
            elif var == 'Q':
                var_data = -Q    
            
            # Initialize vortex objects with PIV coordinates
            # Note: For PIV, we pass x,y coordinates and w velocity (streamwise) for detection
            S_Vortex = vortex_piv('Secondary', SV_WindowLL, SV_WindowUR, x, y, w, var_data, method, level)
            P_Vortex = vortex_piv('Primary', PV_WindowLL, PV_WindowUR, x, y, w, var_data, method, level)
            
            # Check for tertiary vortex based on cut location and availability of window bounds
            if cut_loc != 'PIV1' and TV_WindowLL is not None and TV_WindowUR is not None and len(TV_WindowLL) == 2 and len(TV_WindowUR) == 2:
                flag_tertiary = True
            else:
                flag_tertiary = False
            T_Vortex = vortex_piv('Tertiary', TV_WindowLL, TV_WindowUR, x, y, w, var_data, method, level) if flag_tertiary else None
            
            # Append data
            aggregated_u.append(u)
            aggregated_v.append(v)
            aggregated_w.append(w)
            aggregated_vort.append(vort_x)
            
            # Store core locations - handle empty or invalid core locations
            if len(S_Vortex.core.core_loc) > 0:
                S_core_loc_block.append(S_Vortex.core.core_loc[0])
            else:
                S_core_loc_block.append([np.nan, np.nan])
                
            if len(P_Vortex.core.core_loc) > 0:
                P_core_loc_block.append(P_Vortex.core.core_loc[0])
            else:
                P_core_loc_block.append([np.nan, np.nan])
                
            if T_Vortex and len(T_Vortex.core.core_loc) > 0:
                T_core_loc_block.append(T_Vortex.core.core_loc[0])
            elif T_Vortex:
                T_core_loc_block.append([np.nan, np.nan])
            
        except Exception as e:
            print(f"Block {block_num}: Error processing PIV file {file_path}: {e}")
            continue  # Skip to the next file
    
    # Final progress report for the block
    print(f"    Block {block_num}: Completed processing {total_files_in_block} PIV files")
    
    # Convert lists to numpy arrays
    aggregated_u = np.array(aggregated_u)
    aggregated_v = np.array(aggregated_v)
    aggregated_w = np.array(aggregated_w)
    aggregated_vort = np.array(aggregated_vort)
    S_core_loc_block = np.array(S_core_loc_block)
    P_core_loc_block = np.array(P_core_loc_block)
    T_core_loc_block = np.array(T_core_loc_block) if flag_tertiary else np.array([])
    
    # Return aggregated data as a dictionary
    return {
        'u': aggregated_u,
        'v': aggregated_v,
        'w': aggregated_w,
        'vort_x': aggregated_vort,  # Keep naming consistent with LES for compatibility
        'S_core_loc': S_core_loc_block,
        'P_core_loc': P_core_loc_block,
        'T_core_loc': T_core_loc_block,
        'y': x,  # Map PIV x (wall-normal) to LES y coordinate for compatibility
        'z': y,  # Map PIV y (span) to LES z coordinate for compatibility
        'block_num': block_num
    }


class vortex_piv:
    """
    A PIV-specific vortex class that handles coordinate system differences.
    
    This class adapts the vortex detection for PIV data where:
    - No re-gridding is needed (data is already structured)
    - Coordinate system: x=wall-normal, y=span, z=streamwise
    - Direct detection on the structured PIV grid
    """
    
    def __init__(self, pos: str, A: list, B: list, x, y, w, vort, choice: str, level: float = -20) -> None:
        """
        Initialize PIV vortex detection.
        
        Args:
            pos (str): Vortex position identifier ('Primary', 'Secondary', 'Tertiary')
            A (list): Lower-left corner coordinates [x, y] of detection window (PIV coords)
            B (list): Upper-right corner coordinates [x, y] of detection window (PIV coords)
            x (array-like): X-coordinates (wall-normal) of the PIV grid
            y (array-like): Y-coordinates (span) of the PIV grid
            w (array-like): W-velocity component (streamwise)
            vort (array-like): Vorticity field data (streamwise vorticity)
            choice (str): Detection method ('max', 'precise', 'area')
            level (float, optional): Vorticity threshold for detection. Defaults to -30.
        """
        assert choice in ['max', 'precise', 'area'], "Invalid choice for PIV vortex detection."
        self.position = pos
        self.location = self.vortex_loc(A, B)
        self.core = self.vortex_core_piv(x, y, w, vort, self.location.x, self.location.y, choice, level)
        
    class vortex_loc:
        """
        Spatial boundaries for PIV vortex detection using PIV coordinate system.
        """
        
        def __init__(self, A: list, B: list):
            """
            Initialize spatial boundaries from two corner points.
            
            Args:
                A (list): First corner coordinates [x, y] (PIV coords)
                B (list): Second corner coordinates [x, y] (PIV coords)
            """
            self.x = [np.min([A[0], B[0]]), np.max([A[0], B[0]])]  # Wall-normal bounds
            self.y = [np.min([A[1], B[1]]), np.max([A[1], B[1]])]  # Span bounds
            
    class vortex_core_piv:
        """
        PIV-specific vortex core detection that works directly on structured PIV data.
        """
        
        def __init__(self, x, y, w, vort, x_lim: list, y_lim: list, choice: str, level: int):
            """
            Detect vortex core on PIV structured grid.
            
            Args:
                x (array-like): X-coordinates (wall-normal) of PIV grid
                y (array-like): Y-coordinates (span) of PIV grid  
                w (array-like): W-velocity component (streamwise)
                vort (array-like): Vorticity field data (streamwise vorticity)
                x_lim (list): X-coordinate bounds [min, max] for detection
                y_lim (list): Y-coordinate bounds [min, max] for detection
                choice (str): Detection method ('max', 'precise', 'area')
                level (int): Vorticity threshold for detection
            """
            if choice == 'max':
                # Simple maximum finding within window bounds
                try:
                    mask = (x >= np.min(x_lim)) & (x <= np.max(x_lim)) & (y >= np.min(y_lim)) & (y <= np.max(y_lim))
                    dummy = vort * mask
                    if np.all(dummy == 0) or not np.any(mask):
                        print("All PIV vortex values masked in 'max' choice.")
                        self.core_loc = [[np.nan, np.nan]]
                        self.core_mag = [np.nan]
                    else:
                        valid_indices = np.where(mask)
                        if len(valid_indices[0]) > 0:
                            masked_vort = vort[mask]
                            min_local_idx = np.argmin(masked_vort)
                            min_global_idx = valid_indices[0][min_local_idx]
                            self.core_loc = [[x.flat[min_global_idx], y.flat[min_global_idx]]]
                            self.core_mag = [vort.flat[min_global_idx]]
                        else:
                            self.core_loc = [[np.nan, np.nan]]
                            self.core_mag = [np.nan]
                except Exception as e:
                    print(f"Error in PIV 'max' vortex detection: {e}")
                    self.core_loc = [[np.nan, np.nan]]
                    self.core_mag = [np.nan]
                    
            elif choice == 'precise':
                # For PIV data, work directly on the structured grid
                try:
                    # Relax velocity threshold for PIV data since flow is different
                    mask_w = (np.abs(w) >= 0.1)  # Much lower velocity threshold for PIV
                    mask_vort = (vort <= level) if level <= -5 else (vort >= level)
                    mask_bounds = (x >= np.min(x_lim)) & (x <= np.max(x_lim)) & (y >= np.min(y_lim)) & (y <= np.max(y_lim))
                    combined_mask = mask_w & mask_vort & mask_bounds
                    
                    if not np.any(combined_mask):
                        print("No PIV vortex found in 'precise' choice.")
                        self.core_loc = [[np.nan, np.nan]]
                        self.core_mag = [np.nan]
                    else:
                        valid_indices = np.where(combined_mask)
                        if len(valid_indices[0]) > 0:
                            masked_vort = vort[combined_mask]
                            min_local_idx = np.argmin(masked_vort)
                            min_global_idx = valid_indices[0][min_local_idx]
                            self.core_loc = [[x.flat[min_global_idx], y.flat[min_global_idx]]]
                            self.core_mag = [vort.flat[min_global_idx]]
                        else:
                            self.core_loc = [[np.nan, np.nan]]
                            self.core_mag = [np.nan]
                except Exception as e:
                    print(f"Error in PIV 'precise' vortex detection: {e}")
                    self.core_loc = [[np.nan, np.nan]]
                    self.core_mag = [np.nan]
                    
            elif choice == 'area':
                # Area-based detection on PIV structured grid
                try:
                    # Relax velocity threshold for PIV data  
                    mask_w = (np.abs(w) >= 0.1)  # Much lower velocity threshold for PIV
                    mask_vort = (vort <= level) if level <= -5 else (vort >= level)
                    mask_bounds = (x >= np.min(x_lim)) & (x <= np.max(x_lim)) & (y >= np.min(y_lim)) & (y <= np.max(y_lim))
                    combined_mask = mask_w & mask_vort & mask_bounds
                    
                    if not np.any(combined_mask):
                        print("No PIV vortical area found in 'area' choice.")
                        self.core_loc = [[np.nan, np.nan]]
                        self.core_mag = [np.nan]
                    else:
                        # Find center of mass of vortical region
                        indices = np.where(combined_mask)
                        if len(indices[0]) > 0:
                            # Calculate center of mass
                            center_x = np.mean(x.flat[indices[0]])
                            center_y = np.mean(y.flat[indices[1]]) if len(indices) > 1 else np.mean(y.flat[indices[0]])
                            self.core_loc = [[center_x, center_y]]
                            # Find magnitude at the vortical region
                            masked_vort = vort[combined_mask]
                            self.core_mag = [np.min(masked_vort)]
                        else:
                            self.core_loc = [[np.nan, np.nan]]
                            self.core_mag = [np.nan]
                except Exception as e:
                    print(f"Error in PIV 'area' vortex detection: {e}")
                    self.core_loc = [[np.nan, np.nan]]
                    self.core_mag = [np.nan]


def detect_vortex_piv(source_dir, cut, alpha, var='vort_x', level=-20, method='area', nb_tasks=None, max_file=None, output_dir='./'):
    """
    Detect vortices from PIV data using parallel processing.
    
    This function handles PIV-specific vortex detection without the need for re-gridding
    since PIV data is already structured. It accounts for the different coordinate system
    used in PIV data compared to LES.
    
    Args:
        source_dir (str): Directory path containing PIV HDF5 timestep files (*.h5)
        cut (str): Cut plane identifier (e.g., 'PIV1', 'PIV2', 'PIV3')
        alpha (int): Angle of attack in degrees for window boundary selection
        var (str, optional): Variable name for vortex core tracking (vort_x or lambda2). Defaults to 'vort_x'.
        level (float, optional): Threshold for vortex detection. Defaults to -30.
        method (str, optional): Vortex detection algorithm. Defaults to 'precise'.
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
            - Vars (dict): PIV grid data structure for compatibility
    """
    print('\n----> PIV Data Source Information:')
    print(f'    Source directory: {source_dir}')
    print(f'    Cut location: {cut}')
    
    # Get PIV window boundaries for the given cut location
    try:
        piv_windows = get_window_boundaries_PIV(cut, str(alpha))
    except (KeyError, ValueError) as e:
        print(f"PIV window boundaries not found for cut {cut} at alpha {alpha}: {e}")
        raise

    SV_WindowLL = piv_windows['SV_WindowLL']
    SV_WindowUR = piv_windows['SV_WindowUR']
    PV_WindowLL = piv_windows['PV_WindowLL']
    PV_WindowUR = piv_windows['PV_WindowUR']
    TV_WindowLL = piv_windows['TV_WindowLL']
    TV_WindowUR = piv_windows['TV_WindowUR']

    # Glob all .h5 files in the source directory
    source_files = sorted(glob.glob(os.path.join(source_dir, '*.h5')))
    if max_file is not None:
        source_files = source_files[:max_file]
    
    total_files = len(source_files)
    print(f'    Total number of PIV files to process: {total_files}')

    if total_files == 0:
        print("No .h5 files found in the PIV source directory.")
        raise FileNotFoundError("No .h5 files found in the PIV source directory.")

    # Determine the number of parallel tasks
    avail = cpu_count()
    if nb_tasks is None:
        nb_tasks = avail
    nproc = min(avail, nb_tasks)
    if nproc > total_files:
        nproc = total_files
    
    print('\n----> PIV Parallel Processing Configuration:')
    print(f'    Number of available parallel compute processes: {nproc}')
    print(f'    Number of parallel tasks (blocks): {nproc}')

    # Split the data_files into nproc roughly equal blocks
    file_blocks = np.array_split(source_files, nproc)
    print(f'    PIV data files split into {nproc} blocks.')
    
    # Create blocks with additional parameters for starmap
    blocks = [(file_block, SV_WindowLL, SV_WindowUR, PV_WindowLL, PV_WindowUR, TV_WindowLL, TV_WindowUR, cut, var, level, method, block_num) 
              for block_num, file_block in enumerate(file_blocks)]

    print('\n----> Performing Parallel PIV Vortex Detection...')
    
    # Process blocks in parallel
    with Pool(nproc) as pool:
        results = pool.starmap(process_file_block_piv, blocks)
    
    print('\n----> PIV File Processing Results:')
    print('    PIV file processing complete.')
    
    # Initialize lists to collect results
    aggregated_u, aggregated_v, aggregated_w, aggregated_vort = [], [], [], []
    S_core_loc, P_core_loc, T_core_loc = [], [], []
    y_master, z_master = None, None  # Keep LES naming for compatibility
    
    # Stitch the results together
    for result in results:
        if result['y'] is not None and result['z'] is not None:
            if y_master is None and z_master is None:
                y_master = result['y']  # PIV x mapped to LES y
                z_master = result['z']  # PIV y mapped to LES z
            else:
                if not (np.array_equal(y_master, result['y']) and np.array_equal(z_master, result['z'])):
                    print("Inconsistent PIV coordinates across files. Using the first file's coordinates.")
        
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

    # Check if coordinates are defined
    if y_master is None or z_master is None:
        print("No PIV coordinate data collected from any files.")
        raise ValueError("No PIV coordinate data collected from any files.")
    
    # Calculate time-averaged fields
    mean_vort = np.mean(aggregated_vort, axis=0)
    mean_u = np.mean(aggregated_u, axis=0)
    mean_v = np.mean(aggregated_v, axis=0)
    mean_w = np.mean(aggregated_w, axis=0)
    
    # For PIV, preserve the original 2D grid structure without interpolation
    # PIV data is already on a structured grid - just reshape it properly
    print(f"    PIV data shape: {y_master.shape}, preserving original grid structure...")
    
    # PIV data should already be 2D, but ensure it's properly shaped
    if len(y_master.shape) == 2:
        # Data is already 2D
        grid_y_2d = y_master
        grid_z_2d = z_master
        grid_u_2d = mean_u
        grid_v_2d = mean_v
        grid_w_2d = mean_w
        grid_vort_2d = mean_vort
        
        print(f"    Preserved original grid dimensions: {grid_y_2d.shape}")
    else:
        # Data is 1D - need to determine proper reshaping based on coordinate structure
        # For PIV data, assume it's organized in a regular grid pattern
        # Try to infer the grid dimensions from the coordinate structure
        
        
        unique_y, unique_z = np.sort(np.unique(y_master)), np.sort(np.unique(z_master))
        tol_y, tol_z = np.mean(np.diff(unique_y)), np.mean(np.diff(unique_z))
        dummy_y, dummy_z = [unique_y[0]], [unique_z[0]]
        
        for y in unique_y[1:]:
            if np.abs(y - dummy_y[-1]) > tol_y/2:
                dummy_y.append(y)
        unique_y = np.array(dummy_y)
        
        for z in unique_z[1:]:
            if np.abs(z - dummy_z[-1]) > tol_z/2:
                dummy_z.append(z)
        unique_z = np.array(dummy_z)
        ny, nz = len(unique_y), len(unique_z)
        
        print(f"    Inferred grid dimensions from 1D data: {ny} x {nz}")
        print(f"    Original data length: {len(y_master)}")
        print(f"    Expected reshaped size: {ny * nz}")
        
        # Check if the data can be properly reshaped
        if len(y_master) == ny * nz:
            # Reshape the data to 2D
            grid_y_2d = y_master.reshape(ny, nz, order='F')  # PIV y (wall-normal) to LES y
            grid_z_2d = z_master.reshape(ny, nz, order='F')  # PIV z (span) to LES z
            grid_u_2d = mean_u.reshape(ny, nz, order='F')
            grid_v_2d = mean_v.reshape(ny, nz, order='F')
            grid_w_2d = mean_w.reshape(ny, nz, order='F')
            grid_vort_2d = mean_vort.reshape(ny, nz, order='F')
            
            print(f"    Successfully reshaped to: {grid_y_2d.shape}")
        else:
            print(f"    Warning: PIV data is on sparse grid ({len(y_master)} points vs {ny}x{nz}={ny*nz} full grid)")
            raise ValueError(f"Cannot determine appropriate grid structure for {len(y_master)} points")
    
    class PIVGridData:
        def __init__(self, y_1d, z_1d, u_1d, v_1d, w_1d, vort_1d, 
                     y_2d, z_2d, u_2d, v_2d, w_2d, vort_2d):
            # 2D grid data for plotting
            self.grid_y = y_2d.T  # PIV x (wall-normal) mapped to LES y
            self.grid_z = z_2d.T  # PIV y (span) mapped to LES z
            # Note: grid_u, grid_v, grid_w, grid_vort will be set after airfoil masking
            
            # 1D original data for compatibility
            self.y = y_1d
            self.z = z_1d
            self.u = u_1d
            self.v = v_1d
            self.w = w_1d
            self.vort = vort_1d
            
            # PIV airfoil mask using same logic as LES method
            # Apply airfoil masking: identify regions where |u| < 1e-3 (near-zero velocity)
            index_airf = np.where(
                (np.abs(u_2d) < 1e-5) & 
                (np.abs(v_2d) < 1e-5) & 
                (np.abs(w_2d) < 1e-5) & 
                (np.abs(vort_2d) < 1e-5) &
                (np.arange(ny)[:, None] >= 3) & (np.arange(ny)[:, None] < ny - 3) &
                (np.arange(nz)[None, :] >= 3) & (np.arange(nz)[None, :] < nz - 3)
            )
                        
            # Create copies of 2D grids for masking
            grid_u_masked = u_2d.copy()
            grid_v_masked = v_2d.copy()  
            grid_w_masked = w_2d.copy()
            grid_vort_masked = vort_2d.copy()
            
            # Apply LES masking logic: set velocities to 0 and vorticity to nan at airfoil
            grid_u_masked[index_airf] = 0.0
            grid_v_masked[index_airf] = 0.0
            grid_w_masked[index_airf] = 0.0
            grid_vort_masked[index_airf] = float('nan')
            
            # Update the transposed grid values with masked data
            self.grid_u = grid_u_masked.T
            self.grid_v = grid_v_masked.T
            self.grid_w = grid_w_masked.T
            self.grid_vort = grid_vort_masked.T
            
            # Create mask_indx using same logic as LES: np.flip(np.isnan(grid_vort), axis=0)
            self.mask_indx = np.flip(np.isnan(grid_vort_masked), axis=0)
            

    
    Vars = PIVGridData(y_master, z_master, mean_u, mean_v, mean_w, mean_vort,
                       grid_y_2d, grid_z_2d, grid_u_2d, grid_v_2d, grid_w_2d, grid_vort_2d)
    
    # Calculate mean core locations from instantaneous data (excluding NaN values)
    # Filter out NaN values for mean calculation
    S_core_valid = S_core_loc[~np.isnan(S_core_loc).any(axis=1)]
    P_core_valid = P_core_loc[~np.isnan(P_core_loc).any(axis=1)]
    
    if len(S_core_valid) > 0:
        S_mean_core = [[np.mean(S_core_valid[:, 0]), np.mean(S_core_valid[:, 1])]]
    else:
        S_mean_core = [[np.nan, np.nan]]
        
    if len(P_core_valid) > 0:
        P_mean_core = [[np.mean(P_core_valid[:, 0]), np.mean(P_core_valid[:, 1])]]
    else:
        P_mean_core = [[np.nan, np.nan]]
    
    # Calculate vortex wandering statistics using actual mean of instantaneous data
    S_Vort_Diff = vortex_trace(S_mean_core, S_core_loc)
    P_Vort_Diff = vortex_trace(P_mean_core, P_core_loc)
    
    if tertiary:
        # Calculate mean core location for tertiary vortex
        T_core_valid = T_core_loc[~np.isnan(T_core_loc).any(axis=1)]
        if len(T_core_valid) > 0:
            T_mean_core = [[np.mean(T_core_valid[:, 0]), np.mean(T_core_valid[:, 1])]]
        else:
            T_mean_core = [[np.nan, np.nan]]
        T_Vort_Diff = vortex_trace(T_mean_core, T_core_loc)
    else:
        T_Vort_Diff = [] 
    
    # Saving data using existing save_data function
    print('\n----> Saving PIV Results:')
    print(f'    Saving processed PIV data to: {output_dir}')
    save_data(Vars, cut, P_core_loc, P_Vort_Diff, S_core_loc, S_Vort_Diff, T_core_loc, T_Vort_Diff, output_dir, tertiary, data_type='LES')
    print('    PIV data saved successfully')
    
    return S_core_loc, S_Vort_Diff, P_core_loc, P_Vort_Diff, T_core_loc, T_Vort_Diff, Vars