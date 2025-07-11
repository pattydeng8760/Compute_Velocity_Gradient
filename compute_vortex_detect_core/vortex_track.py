
import numpy as np
from itertools import combinations
from .make_grid import make_grid
from .utils import print_custom

# Define the vortex-related classes
class vortex:
    """
    A class representing a vortex detected in the flow field.
    
    This class encapsulates vortex detection and core identification functionality.
    It supports multiple detection methods and can identify vortex cores within
    specified spatial windows.
    
    Attributes:
        position (str): Identifier for the vortex (e.g., 'Primary', 'Secondary', 'Tertiary')
        location (vortex_loc): Spatial boundaries for vortex detection
        core (vortex_core): Detected vortex core location and magnitude
    
    Methods:
        __init__: Initialize vortex detection with specified parameters
    """
    
    def __init__(self, pos: str, A: list, B: list, y, z, u, vort, choice: str, level: float = -30) -> None:
        """
        Initialize vortex detection and core identification.
        
        Args:
            pos (str): Vortex position identifier ('Primary', 'Secondary', 'Tertiary')
            A (list): Lower-left corner coordinates [y, z] of detection window
            B (list): Upper-right corner coordinates [y, z] of detection window
            y (array-like): Y-coordinates of the flow field grid
            z (array-like): Z-coordinates of the flow field grid
            u (array-like): Velocity component in x-direction
            vort (array-like): Vorticity field data
            choice (str): Detection method ('max', 'precise', 'area')
                - 'max': Find maximum vorticity point in window
                - 'precise': Use gridding and connected component analysis
                - 'area': Find center of largest vortical region
            level (float, optional): Vorticity threshold for detection. Defaults to -30.
        
        Raises:
            AssertionError: If choice is not one of the valid detection methods
        """
        assert choice in ['max', 'precise', 'area'], "Invalid choice for vortex detection."
        self.position = pos
        self.location = self.vortex_loc(A, B)
        self.core = self.vortex_core(y, z, u, vort, self.location.y, self.location.z, choice, level)
        
    class vortex_loc:
        """
        A nested class defining the spatial boundaries for vortex detection.
        
        Attributes:
            y (list): Y-coordinate bounds [min, max] for the detection window
            z (list): Z-coordinate bounds [min, max] for the detection window
        """
        
        def __init__(self, A: list, B: list):
            """
            Initialize spatial boundaries from two corner points.
            
            Args:
                A (list): First corner coordinates [y, z]
                B (list): Second corner coordinates [y, z]
            """
            self.y = [np.min([A[0], B[0]]), np.max([A[0], B[0]])]
            self.z = [np.min([A[1], B[1]]), np.max([A[1], B[1]])]
            
    class vortex_core:
        """
        A nested class for detecting and locating vortex cores.
        
        This class implements different algorithms for vortex core detection
        including simple maximum finding, precise gridding with connected component
        analysis, and area-based center finding.
        
        Attributes:
            core_loc (list): List of [y, z] coordinates of detected vortex cores
            core_mag (list): List of vorticity magnitudes at detected core locations
        """
        
        def __init__(self, y, z, u, vort, y_lim: list, z_lim: list, choice: str, level: int):
            """
            Detect vortex core using the specified method.
            
            Args:
                y (array-like): Y-coordinates of the flow field
                z (array-like): Z-coordinates of the flow field  
                u (array-like): Velocity component in x-direction
                vort (array-like): Vorticity field data
                y_lim (list): Y-coordinate bounds [min, max] for detection
                z_lim (list): Z-coordinate bounds [min, max] for detection
                choice (str): Detection method ('max', 'precise', 'area')
                level (int): Vorticity threshold for detection
            """
            if choice == 'max':
                mask = (y >= np.min(y_lim)) & (y <= np.max(y_lim)) & (z >= np.min(z_lim)) & (z <= np.max(z_lim))
                dummy = vort * mask
                if np.all(dummy == 0):
                    print_custom("All vortex values masked in 'max' choice.")
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
                    print_custom("No largest area found in 'precise' choice.")
                    self.core_loc = [[np.nan, np.nan]]
                    self.core_mag = [np.nan]
                else:
                    mask3 = np.zeros_like(mask1)
                    mask3[largest_area[0][0]:largest_area[1][0], largest_area[0][1]:largest_area[1][1]] = 1
                    mask = mask3 & mask2
                    dummy = vars.grid_vort * mask
                    index = np.argwhere(dummy == np.min(dummy))
                    if index.size == 0:
                        print_custom("No minimum found in 'precise' choice for vortex core.")
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
                    print_custom("No largest area found in 'area' choice.")
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
    """
    A class for calculating vortex wandering statistics.
    
    This class computes the spatial deviation of instantaneous vortex core locations
    from their time-averaged position, providing metrics for vortex wandering behavior.
    
    Attributes:
        diff (numpy.ndarray): Array of distances between instantaneous and mean positions
    """
    
    def __init__(self, mean, inst):
        """
        Initialize vortex trace calculation.
        
        Args:
            mean (list): Time-averaged vortex core location [[y_mean, z_mean]]
            inst (array-like): Array of instantaneous vortex core locations [[y1, z1], [y2, z2], ...]
        """
        self.calculate_trace(mean, inst)
        
    def calculate_trace(self, mean, inst):
        """
        Calculate the Euclidean distance between instantaneous and mean positions.
        
        This method computes the wandering amplitude for each time step as the 
        distance from the time-averaged vortex core location.
        
        Args:
            mean (list): Time-averaged vortex core location [[y_mean, z_mean]]
            inst (array-like): Array of instantaneous vortex core locations
        """
        diff = np.zeros((np.shape(inst)[0], 1))
        for i in range(np.shape(inst)[0]):
            diff[i] = np.sqrt((mean[0][0] - inst[i][0])**2 + (mean[0][1] - inst[i][1])**2) 
        self.diff = diff

def find_squares(a):
    """
    Find connected components in a binary 2D array and identify the largest region.
    
    This function performs connected component analysis on a binary array to identify
    contiguous regions of True values. It uses 8-connectivity (including diagonal
    neighbors) and returns information about all connected components, with special
    emphasis on the largest component.
    
    Args:
        a (numpy.ndarray): 2D binary array where True/1 values represent regions of interest
    
    Returns:
        tuple: A tuple containing:
            - bounds (list): List of bounding boxes for each component as ((min_i, min_j), (max_i, max_j))
            - size_bounds (list): List of areas (width × height) for each component's bounding box
            - largest_area (tuple or list): Bounding box of the largest component, or empty list if none found
    
    Example:
        >>> import numpy as np
        >>> binary_array = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
        >>> bounds, sizes, largest = find_squares(binary_array)
        >>> print_custom(f"Found {len(bounds)} components with sizes {sizes}")
    """
    # Find all positions with True/1 values
    ones = [(i, j) for i, row in enumerate(a) for j, val in enumerate(row) if val]
    if not ones:
        print_custom("No connected squares found.")
        return [], [], []
    
    # Build adjacency graph using 8-connectivity (including diagonals)
    graph = {a: [] for a in ones}
    for a, b in combinations(ones, 2):
        if abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1:
            graph[a].append(b)
            graph[b].append(a)
    
    # Find connected components using depth-first search
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
    
    # Calculate bounding boxes for each component
    bounds = [((min(a[0] for a in c), min(a[1] for a in c)),
               (max(a[0] for a in c), max(a[1] for a in c)))
              for c in components]
    
    # Calculate bounding box areas
    size_bounds = [(bounds[i][1][1] - bounds[i][0][1]) * (bounds[i][1][0] - bounds[i][0][0]) 
                   for i in range(len(bounds))]
    
    if not size_bounds:
        print_custom("No size bounds calculated.")
        largest_area = []
    else:
        # Return the bounding box of the component with the largest area
        largest_area = bounds[np.argmax(size_bounds)]
    
    return bounds, size_bounds, largest_area