import numpy as np
from itertools import combinations
from scipy.ndimage import binary_dilation
from .grid_maker import make_grid
from .utils import print

class vortex:
    """Class for detecting and tracking vortex cores."""
    
    def __init__(self, pos, A, B, y, z, u, vort, choice, level):
        """
        Initialize vortex detection.
        
        Parameters:
        -----------
        pos : str
            Position identifier
        A : list
            Lower bound coordinates [y, z]
        B : list
            Upper bound coordinates [y, z]
        y : array
            Y coordinates
        z : array
            Z coordinates
        u : array
            Velocity magnitude
        vort : array
            Vorticity values
        choice : str
            Detection method ('max', 'precise', 'area')
        level : int
            Threshold level for detection
        """
        assert choice in ['max', 'precise', 'area'], "Invalid choice for vortex detection."
        
        self.position = pos
        self.location = self.vortex_loc(A, B)
        self.core = self.vortex_core(y, z, u, vort, self.location.y, self.location.z, choice, level)
        
        print(f"        Detected {pos} vortex at: {self.core.core_loc}")
    
    class vortex_loc:
        """Define vortex location boundaries."""
        
        def __init__(self, A, B):
            self.y = [min(A[0], B[0]), max(A[0], B[0])]
            self.z = [min(A[1], B[1]), max(A[1], B[1])]
    
    class vortex_core:
        """Detect vortex core using specified method."""
        
        def __init__(self, y, z, u, vort, y_lim, z_lim, choice, level):
            self.detect_core(y, z, u, vort, y_lim, z_lim, choice, level)
        
        def detect_core(self, y, z, u, vort, y_lim, z_lim, choice, level):
            """Detect vortex core using specified method."""
            if choice == 'max':
                self._detect_max(y, z, u, vort, y_lim, z_lim)
            elif choice == 'precise':
                self._detect_precise(y, z, u, vort, y_lim, z_lim, level)
            elif choice == 'area':
                self._detect_area(y, z, u, vort, y_lim, z_lim, level)
        
        def _detect_max(self, y, z, u, vort, y_lim, z_lim):
            """Detect vortex core using maximum method."""
            mask = ((y >= min(y_lim)) & (y <= max(y_lim)) & 
                   (z >= min(z_lim)) & (z <= max(z_lim)))
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
            """Detect vortex core using precise method."""
            n = 200
            vars = make_grid(n, y_lim, z_lim, y, z, u, vort, 'grid_vort', True)
            
            mask1 = (vars.grid_vort <= level) if level <= -10 else (vars.grid_vort >= level)
            _, _, largest_area = find_squares(mask1)
            
            if not largest_area:
                print("            No largest area found in 'precise' choice.")
                self.core_loc = [[np.nan, np.nan]]
                self.core_mag = [np.nan]
                return
            
            # Create mask for largest area
            mask2 = np.zeros_like(mask1)
            mask2[largest_area[0][0]:largest_area[1][0],
                  largest_area[0][1]:largest_area[1][1]] = 1
            mask = mask1 & mask2
            
            # Handle airfoil masking
            bool_array = np.abs(vars.u) < 5
            if np.sum(bool_array) > 0.05 * len(bool_array.flatten()):
                mask[bool_array] = 0
                dilated = binary_dilation(bool_array)
                boundary = dilated & ~bool_array
                boundary_indices = np.argwhere(boundary)
                
                if len(boundary_indices) > 0:
                    min_y, min_z = np.min(boundary_indices, axis=0)
                    min_y, min_z = max(0, min_y - 10), max(0, min_z - 10)
                    max_y, max_z = np.max(boundary_indices, axis=0)
                    max_y = min(vars.grid_vort.shape[0], max_y + 10)
                    max_z = min(vars.grid_vort.shape[1], max_z + 10)
                    mask[min_y:max_y, min_z:max_z] = 0
            
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
                self.core_mag = [dummy[index[0, 0], index[0, 1]]]
            
            del vars
        
        def _detect_area(self, y, z, u, vort, y_lim, z_lim, level):
            """Detect vortex core using area method."""
            n = 200
            vars = make_grid(n, y_lim, z_lim, y, z, u, vort, 'grid_vort', True)
            
            mask1 = (vars.grid_vort <= level) if level <= -10 else (vars.grid_vort >= level)
            _, _, largest_area = find_squares(mask1)
            
            if not largest_area:
                print("            No largest area found in 'area' choice.")
                self.core_loc = [[np.nan, np.nan]]
                self.core_mag = [np.nan]
                return
            
            # Use center of largest area
            mask2 = np.zeros_like(mask1)
            mask2[largest_area[0][0]:largest_area[1][0],
                  largest_area[0][1]:largest_area[1][1]] = 1
            mask = mask1 & mask2
            dummy = vars.grid_vort * mask
            
            # Find center of area
            index = [(largest_area[0][0] + largest_area[1][0]) / 2,
                     (largest_area[0][1] + largest_area[1][1]) / 2]
            y_core = vars.grid_y[int(index[0]), int(index[1])]
            z_core = vars.grid_z[int(index[0]), int(index[1])]
            self.core_loc = [[y_core, z_core]]
            self.core_mag = [np.nanmin(dummy) if not np.isnan(dummy).all() else np.nan]
            
            del vars

def find_squares(a):
    """Find connected squares in a boolean array."""
    ones = [(i, j) for i, row in enumerate(a) for j, val in enumerate(row) if val]
    
    if not ones:
        print("            No connected squares found.")
        return [], [], []
    
    # Build graph of connected points
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
    
    # Calculate bounds and sizes
    bounds = [((min(pt[0] for pt in comp), min(pt[1] for pt in comp)),
               (max(pt[0] for pt in comp), max(pt[1] for pt in comp)))
              for comp in components]
    
    size_bounds = [(b[1][0]-b[0][0]) * (b[1][1]-b[0][1]) for b in bounds]
    largest_area = bounds[np.argmax(size_bounds)] if size_bounds else []
    
    return bounds, size_bounds, largest_area