import numpy as np
from scipy.interpolate import griddata
from .utils import print

class make_grid:
    """Class for creating interpolated grids from scattered data."""
    
    def __init__(self, n, y_bnd, z_bnd, y, z, u, var, name, airfoil=True):
        """
        Initialize grid and interpolate initial variables.
        
        Parameters:
        -----------
        n : int
            Number of grid points in each direction
        y_bnd : list
            Y-axis boundaries [min, max]
        z_bnd : list
            Z-axis boundaries [min, max]
        y : array
            Y coordinates of scattered data
        z : array
            Z coordinates of scattered data
        u : array
            U velocity component
        var : array
            Variable to interpolate
        name : str
            Name of the variable
        airfoil : bool
            Whether to consider airfoil masking
        """
        # Create linear grids
        y_lin = np.linspace(min(y_bnd), max(y_bnd), num=n)
        z_lin = np.linspace(min(z_bnd), max(z_bnd), num=n)
        self.grid_y, self.grid_z = np.meshgrid(y_lin, z_lin)
        
        # Store original coordinates
        self.y, self.z = y, z
        
        # Interpolate velocity
        self.u = griddata(np.transpose([self.y, self.z]), u, (self.grid_y, self.grid_z), method='linear')
        
        # Handle airfoil masking
        self.airfoil = airfoil
        if self.airfoil and len(np.where(np.abs(self.u) < 1e-3)[0]) < 0.05 * len(self.u.flatten()):
            self.airfoil = False
        
        if self.airfoil:
            self.index_airf = np.where(np.abs(self.u) < 1e-3)
            self.u[self.index_airf] = 0
            self.mask_y = [np.min(self.grid_y[self.index_airf]), np.max(self.grid_y[self.index_airf])]
            self.mask_z = [np.min(self.grid_z[self.index_airf]), np.max(self.grid_z[self.index_airf])]
            self.mask_indx = np.flip(np.isnan(self.u), axis=0)
        
        # Calculate initial variable
        self.calculate_grid(n, y_bnd, z_bnd, var, name)
    
    def calculate_grid(self, n, y_bnd, z_bnd, var, name):
        """
        Calculate and interpolate a variable onto the grid.
        
        Parameters:
        -----------
        n : int
            Number of grid points
        y_bnd : list
            Y-axis boundaries
        z_bnd : list
            Z-axis boundaries
        var : array
            Variable to interpolate
        name : str
            Variable name
        """
        print(f"        Interpolating variable: {name}")
        
        if len(var) == 0:
            interpolated_var = []
        else:
            interpolated_var = griddata(
                np.transpose([self.y, self.z]), var, 
                (self.grid_y, self.grid_z), method='linear'
            )
        
        # Set as attribute
        setattr(self, name, interpolated_var)
        
        # Apply airfoil masking if applicable
        if self.airfoil and len(interpolated_var) > 0:
            getattr(self, name)[self.index_airf] = float('nan')
    
    def get_grid_coordinates(self):
        """Return grid coordinates."""
        return self.grid_y, self.grid_z
    
    def get_grid_variable(self, name):
        """Get interpolated variable by name."""
        return getattr(self, name, None)
    
    def mask_airfoil_region(self, variable):
        """Apply airfoil masking to a variable."""
        if self.airfoil:
            variable[self.index_airf] = float('nan')
        return variable