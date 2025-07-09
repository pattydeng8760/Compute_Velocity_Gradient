



# Define the vortex-related classes
class vortex:
    """
    This class represents a vortex detected in the flow field.
    It takes in the 
    
    
    
    """
    
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