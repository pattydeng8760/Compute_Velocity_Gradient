import numpy as np
from .plotter import plot_local_invariants_QR, plot_local_invariants_Qs_Rs, plot_local_invariants_Qs_Qw
from .data_saver import save_extracted_data
from .utils import print

def find_closest_indices_and_adjacent_cells(core_loc, grid, connectivity, n=6, radius=0.01,
                                            start_angle=0, end_angle=180, n_layers=1):
    """
    Find closest grid indices and adjacent cells around vortex core.
    
    Parameters:
    -----------
    core_loc : list
        Core location [y, z]
    grid : list
        Grid coordinates [y, z]
    connectivity : array
        Connectivity matrix
    n : int
        Number of points around core
    radius : float
        Radius for point distribution
    start_angle : float
        Starting angle in degrees
    end_angle : float
        Ending angle in degrees
    n_layers : int
        Number of adjacent cell layers
    
    Returns:
    --------
    tuple
        Location points, closest indices, adjacent points list
    """
    # Create points around core in circular pattern
    angles = np.linspace(start_angle * np.pi / 180, end_angle * np.pi / 180, n - 1, endpoint=True)
    x = core_loc[0] + radius * np.cos(angles)
    y = core_loc[1] + radius * np.sin(angles)
    x = np.insert(x, 0, core_loc[0])
    y = np.insert(y, 0, core_loc[1])
    loc_points = np.array([x, y])
    
    closest_indices = []
    adjacent_points_list = []
    
    for i in range(loc_points.shape[1]):
        # Find closest grid point
        dist = np.sqrt((grid[0] - loc_points[0, i])**2 + (grid[1] - loc_points[1, i])**2)
        closest_index = np.argmin(dist)
        closest_indices.append(closest_index)
        
        # Find adjacent cells using connectivity
        rows = np.where(np.any(connectivity == closest_index, axis=1))[0]
        if len(rows) == 0:
            adjacent = set()
        else:
            chosen_row = connectivity[rows[0], :]
            adjacent = set(chosen_row)
            adjacent.discard(closest_index)
        
        # Add multiple layers of adjacent cells
        for layer in range(1, n_layers):
            new_adjacent = set()
            for pt in adjacent:
                rows_pt = np.where(np.any(connectivity == pt, axis=1))[0]
                if len(rows_pt) > 0:
                    chosen_row_pt = connectivity[rows_pt[0], :]
                    new_adjacent.update(chosen_row_pt)
            adjacent.update(new_adjacent)
            adjacent.discard(closest_index)
        
        adjacent_points_list.append(adjacent)
    
    return loc_points, closest_indices, adjacent_points_list

def extract_variable_data_stacked(variable_all, core_indices, adjacent_points_list):
    """
    Extract and stack variable data from core and adjacent cells.
    
    Parameters:
    -----------
    variable_all : array
        Variable data with shape (n_cells, n_time)
    core_indices : list
        Core cell indices
    adjacent_points_list : list
        List of adjacent cell sets
    
    Returns:
    --------
    array
        Stacked variable data
    """
    T = variable_all.shape[1]
    stacked_list = []
    
    for core, adj_set in zip(core_indices, adjacent_points_list):
        cell_indices = [core] + sorted(list(adj_set))
        data = variable_all[cell_indices, :]
        stacked = data.reshape(-1)
        stacked_list.append(stacked)
    
    return np.vstack(stacked_list)

def extract_mean_variable(variable_all, core_indices, adjacent_points_list):
    """
    Extract mean variable values from core and adjacent cells.
    
    Parameters:
    -----------
    variable_all : array
        Variable data
    core_indices : list
        Core cell indices
    adjacent_points_list : list
        List of adjacent cell sets
    
    Returns:
    --------
    array
        Mean variable values
    """
    averaged_list = []
    
    for core, adj_set in zip(core_indices, adjacent_points_list):
        cell_indices = [core] + sorted(list(adj_set))
        data = variable_all[cell_indices]
        avg_value = np.mean(data)
        averaged_list.append(avg_value)
    
    return np.array(averaged_list)

def extract_velocity_invariants(data, connectivity, Vortex, location: str, Vortex_Type: str, 
                               radius=0.01, n=6, n_layers=2, start_angle=0, end_angle=180, 
                               data_type: str = 'LES'):
    """
    Extract velocity invariants at vortex core and adjacent cells.
    
    Parameters:
    -----------
    data : dict
        Velocity invariant data
    connectivity : array
        Connectivity matrix
    Vortex : object
        Vortex object with core location
    location : str
        Location identifier
    Vortex_Type : str
        Vortex type identifier
    radius : float
        Radius for point distribution
    n : int
        Number of points around core
    n_layers : int
        Number of adjacent cell layers
    start_angle : float
        Starting angle in degrees
    end_angle : float
        Ending angle in degrees
    data_type : str
        Data type ('LES' or 'PIV')
    
    Returns:
    --------
    array
        Location points used for extraction
    """
    print("        Processing data for {} vortex at location: {}".format(Vortex_Type, location))
    
    grid = [data['y'], data['z']]
    
    # Find closest indices and adjacent cells
    loc_points, closest_indices, adjacent_points_list = find_closest_indices_and_adjacent_cells(
        Vortex.core.core_loc[0], grid, connectivity, 
        n=n, radius=radius, n_layers=n_layers, 
        start_angle=start_angle, end_angle=end_angle
    )
    
    # Extract invariant data
    Phat = extract_variable_data_stacked(data['Phat_all'], closest_indices, adjacent_points_list)
    Qhat = extract_variable_data_stacked(data['Qhat_all'], closest_indices, adjacent_points_list)
    Rhat = extract_variable_data_stacked(data['Rhat_all'], closest_indices, adjacent_points_list)
    Rs = extract_variable_data_stacked(data['Rs_all'], closest_indices, adjacent_points_list)
    Qs = extract_variable_data_stacked(data['Qs_all'], closest_indices, adjacent_points_list)
    Qw = extract_variable_data_stacked(data['Qw_all'], closest_indices, adjacent_points_list)
    
    # Extract mean values for normalization
    var_A = extract_mean_variable(data['var_A'], closest_indices, adjacent_points_list)
    var_S = extract_mean_variable(data['var_S'], closest_indices, adjacent_points_list)
    var_omega = extract_mean_variable(data['var_omega'], closest_indices, adjacent_points_list)
    mean_SR = extract_mean_variable(data['mean_SR'], closest_indices, adjacent_points_list)
    
    # Normalize strain and rotation invariants
    Rs = Rs / (var_S[:, np.newaxis] ** (3/2))
    Qs = Qs / (var_S[:, np.newaxis])
    Qw = Qw / (var_S[:, np.newaxis])
    
    # Generate local invariant plots
    plot_local_invariants_QR(location, Rhat, Qhat, Vortex_Type, data_type)
    plot_local_invariants_Qs_Rs(location, Rs, Qs, Vortex_Type, data_type)
    plot_local_invariants_Qs_Qw(location, Qw, Qs, Vortex_Type, data_type)
    
    # Save extracted data
    save_extracted_data(location, Phat, Qhat, Rhat, Qs, Qw, Rs, Vortex_Type)
    
    return loc_points