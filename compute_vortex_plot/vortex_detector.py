import numpy as np
from scipy.ndimage import binary_dilation
from itertools import combinations
from .grid_maker import make_grid
from .utils import print

class vortex:
    def __init__(self, pos: str, A: list, B: list, y, z, u, vort, choice: str, level: int) -> None:
        """
        Initialize vortex detection class.
        
        Parameters:
        -----------
        pos : str
            Vortex position identifier
        A : list
            Lower left corner of search window
        B : list
            Upper right corner of search window
        y : array
            Y coordinates
        z : array
            Z coordinates
        u : array
            Velocity field
        vort : array
            Vorticity field
        choice : str
            Detection method ('max', 'precise', 'area')
        level : int
            Vorticity threshold level
        """
        assert choice in ['max', 'precise', 'area'], "Invalid choice for vortex detection."
        self.position = pos
        self.location = self.vortex_loc(A, B)
        self.core = self.vortex_core(y, z, u, vort, self.location.y, self.location.z, choice, level)
        
    class vortex_loc:
        def __init__(self, A: list, B: list):
            """Define vortex search location boundaries."""
            self.y = [min(A[0], B[0]), max(A[0], B[0])]
            self.z = [min(A[1], B[1]), max(A[1], B[1])]
            
    class vortex_core:
        def __init__(self, y, z, u, vort, y_lim: list, z_lim: list, choice: str, level: int):
            """
            Detect vortex core using specified method.
            
            Parameters:
            -----------
            y : array
                Y coordinates
            z : array
                Z coordinates
            u : array
                Velocity field
            vort : array
                Vorticity field
            y_lim : list
                Y limits for search window
            z_lim : list
                Z limits for search window
            choice : str
                Detection method
            level : int
                Vorticity threshold
            """
            if choice == 'max':
                self._detect_max(y, z, u, vort, y_lim, z_lim)
            elif choice == 'precise':
                self._detect_precise(y, z, u, vort, y_lim, z_lim, level)
            elif choice == 'area':
                self._detect_area(y, z, u, vort, y_lim, z_lim, level)
        
        def _detect_max(self, y, z, u, vort, y_lim, z_lim):
            """Detect vortex core using maximum vorticity method."""
            mask = (y >= min(y_lim)) & (y <= max(y_lim)) & (z >= min(z_lim)) & (z <= max(z_lim))
            dummy = vort * mask
            
            if np.all(dummy == 0):
                print("            All vortex values masked in 'max' choice.")
                self.core_loc = [[np.nan, np.nan]]
                self.core_mag = [np.nan]
            else:
                min_idx = np.argmin(dummy)
                self.core_loc = [[y[min_idx], z[min_idx]]]
                self.core_mag = [vort[min_idx]]
        
        def _detect_precise(self, y, z, u, vort, y_lim, z_lim, level):
            """Detect vortex core using precise method with grid interpolation."""
            n = 200
            vars = make_grid(n, y_lim, z_lim, y, z, u, vort, 'grid_vort', True)
            
            # Apply vorticity threshold
            mask1 = (vars.grid_vort <= level) if level <= -10 else (vars.grid_vort >= level)
            
            # Find largest connected area
            _, _, largest_area = find_squares(mask1)
            
            if not largest_area:
                print("            No largest area found in 'precise' choice.")
                self.core_loc = [[np.nan, np.nan]]
                self.core_mag = [np.nan]
            else:
                mask2 = np.zeros_like(mask1)
                mask2[largest_area[0][0]:largest_area[1][0],
                      largest_area[0][1]:largest_area[1][1]] = 1
                mask = mask1 & mask2
                
                # Apply airfoil masking
                bool_array = np.abs(vars.u) < 5
                if np.sum(bool_array) > 0.05 * len(bool_array):
                    mask[bool_array] = 0
                    dilated = binary_dilation(bool_array)
                    boundary = dilated & ~bool_array
                    boundary_indices = np.argwhere(boundary)
                    min_y, min_z = np.min(boundary_indices, axis=0)
                    min_y, min_z = max(0, min_y - 10), max(0, min_z - 10)
                    max_y, max_z = np.max(boundary_indices, axis=0)
                    max_y, max_z = min(vars.grid_vort.shape[0], max_y + 10), min(vars.grid_vort.shape[1], max_z + 10)
                    mask[min_y:max_y, min_z:max_z] = 0
                
                # Find minimum vorticity location
                dummy = vars.grid_vort * mask
                index = np.argwhere(dummy == np.nanmin(dummy))
                
                if index.size == 0:
                    print("            No minimum found in 'precise' choice for vortex core.")
                    self.core_loc = [[np.nan, np.nan]]
                    self.core_mag = [np.nan]
                else:
                    y_core = vars.grid_y[index[0, 0], index[0, 1]]
                    z_core = vars.grid_z[index[0, 0], index[0, 1]]
                    self.core_loc = [[y_core, z_core]]
                    self.core_mag = [vort[np.argmin(dummy)]]
            
            del vars
        
        def _detect_area(self, y, z, u, vort, y_lim, z_lim, level):
            """Detect vortex core using area-based method."""
            n = 200
            vars = make_grid(n, y_lim, z_lim, y, z, u, vort, 'grid_vort', True)
            
            # Apply vorticity threshold
            mask1 = (vars.grid_vort <= level) if level <= -10 else (vars.grid_vort >= level)
            
            # Find largest connected area
            _, _, largest_area = find_squares(mask1)
            
            if not largest_area:
                print("            No largest area found in 'area' choice.")
                self.core_loc = [[np.nan, np.nan]]
                self.core_mag = [np.nan]
            else:
                mask2 = np.zeros_like(mask1)
                mask2[largest_area[0][0]:largest_area[1][0],
                      largest_area[0][1]:largest_area[1][1]] = 1
                mask = mask1 & mask2
                dummy = vars.grid_vort * mask
                
                # Use center of largest area as core location
                index = [(largest_area[0][0] + largest_area[1][0]) / 2,
                         (largest_area[0][1] + largest_area[1][1]) / 2]
                y_core = vars.grid_y[int(index[0]), int(index[1])]
                z_core = vars.grid_z[int(index[0]), int(index[1])]
                self.core_loc = [[y_core, z_core]]
                self.core_mag = [vort[np.argmin(dummy)]]
            
            del vars

def find_squares(a):
    """Find connected components in a binary array."""
    ones = [(i, j) for i, row in enumerate(a) for j, val in enumerate(row) if val]
    
    if not ones:
        print("            No connected squares found.")
        return [], [], []
    
    # Build adjacency graph
    graph = {pt: [] for pt in ones}
    for pt1, pt2 in combinations(ones, 2):
        if abs(pt1[0] - pt2[0]) <= 1 and abs(pt1[1] - pt2[1]) <= 1:
            graph[pt1].append(pt2)
            graph[pt2].append(pt1)
    
    # Find connected components
    components = []
    for pt, neighbors in graph.items():
        if any(pt in comp for comp in components):
            continue
        
        comp = set()
        comp.add(pt)
        pending = neighbors.copy()
        
        while pending:
            p = pending.pop()
            if p not in comp:
                comp.add(p)
                pending.extend(graph[p])
        
        components.append(comp)
    
    # Calculate bounding boxes and areas
    bounds = [((min(pt[0] for pt in comp), min(pt[1] for pt in comp)),
               (max(pt[0] for pt in comp), max(pt[1] for pt in comp)))
              for comp in components]
    
    size_bounds = [(b[1][0]-b[0][0]) * (b[1][1]-b[0][1]) for b in bounds]
    largest_area = bounds[np.argmax(size_bounds)] if size_bounds else []
    
    return bounds, size_bounds, largest_area