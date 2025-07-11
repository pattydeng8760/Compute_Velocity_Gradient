import numpy as np
from scipy.interpolate import griddata
from .utils import print

class make_grid:
    def __init__(self, n: int, y_bnd: list, z_bnd: list, y, z, u, var, name: str, airfoil: bool = True):
        """
        Initialize grid interpolation class.
        
        Parameters:
        -----------
        n : int
            Grid resolution
        y_bnd : list
            Y-axis boundaries [min, max]
        z_bnd : list
            Z-axis boundaries [min, max]
        y : array
            Original y coordinates
        z : array
            Original z coordinates
        u : array
            Velocity field for airfoil detection
        var : array
            Variable to interpolate
        name : str
            Variable name
        airfoil : bool
            Whether to detect and mask airfoil region
        """
        y_lin = np.linspace(min(y_bnd), max(y_bnd), num=n)
        z_lin = np.linspace(min(z_bnd), max(z_bnd), num=n)
        self.grid_y, self.grid_z = np.meshgrid(y_lin, z_lin)
        self.y, self.z = y, z
        self.u = griddata(np.transpose([self.y, self.z]), u, (self.grid_y, self.grid_z), method='linear')
        self.airfoil = airfoil
        
        # Check if airfoil region exists
        if self.airfoil and len(np.where(np.abs(self.u) < 1e-3)[0]) < 0.05 * len(self.u):
            self.airfoil = False
        
        # Set up airfoil masking
        if self.airfoil:
            self.index_airf = np.where(np.abs(self.u) < 1e-3)
            self.u[self.index_airf] = 0
            self.mask_y = [np.min(self.grid_y[self.index_airf]), np.max(self.grid_y[self.index_airf])]
            self.mask_z = [np.min(self.grid_z[self.index_airf]), np.max(self.grid_z[self.index_airf])]
            self.mask_indx = np.flip(np.isnan(self.u), axis=0)
        
        # Calculate grid for the initial variable
        self.calculate_grid(n, y_bnd, z_bnd, var, name)
        
    def calculate_grid(self, n: int, y_bnd: list, z_bnd: list, var, name: str):
        """Calculate interpolated grid for a variable."""
        print("        Interpolating variable: {}".format(name))
        
        if len(var) == 0:
            interpolated_var = []
        else:
            interpolated_var = griddata(np.transpose([self.y, self.z]),
                                      var, (self.grid_y, self.grid_z), method='linear')
        
        setattr(self, name, interpolated_var)
        
        # Apply airfoil mask if applicable
        if self.airfoil and len(var) > 0:
            getattr(self, name)[self.index_airf] = float('nan')