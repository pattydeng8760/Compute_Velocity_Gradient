
import numpy as np
from scipy.interpolate import griddata

class make_grid:
    """
    A class for interpolating unstructured flow field data onto a structured rectangular grid.
    
    This class performs linear interpolation to convert scattered CFD/PIV data points onto
    a regular Cartesian grid suitable for visualization and analysis. It supports airfoil
    surface masking and handles multiple flow variables simultaneously.
    
    Attributes:
        grid_y (numpy.ndarray): 2D array of y-coordinates for the structured grid
        grid_z (numpy.ndarray): 2D array of z-coordinates for the structured grid
        grid_u (numpy.ndarray): Interpolated u-velocity component on the grid
        grid_v (numpy.ndarray): Interpolated v-velocity component on the grid
        grid_w (numpy.ndarray): Interpolated w-velocity component on the grid
        grid_vort (numpy.ndarray): Interpolated vorticity field on the grid
        y (array-like): Original unstructured y-coordinates
        z (array-like): Original unstructured z-coordinates
        u (array-like): Original unstructured u-velocity data
        v (array-like): Original unstructured v-velocity data
        w (array-like): Original unstructured w-velocity data
        vort (array-like): Original unstructured vorticity data
        mask_y (list, optional): Y-coordinate bounds of airfoil surface
        mask_z (list, optional): Z-coordinate bounds of airfoil surface
        mask_indx (numpy.ndarray, optional): Boolean mask for airfoil surface regions
    """
    
    def __init__(self, n: int, y_bnd: list, z_bnd: list, y, z, u, v, w, vort, airfoil: bool):
        """
        Initialize grid interpolation.
        
        Args:
            n (int): Number of grid points in each direction (creates n×n grid)
            y_bnd (list): Y-coordinate boundaries [y_min, y_max] for the grid
            z_bnd (list): Z-coordinate boundaries [z_min, z_max] for the grid
            y (array-like): Unstructured y-coordinates of data points
            z (array-like): Unstructured z-coordinates of data points
            u (array-like): U-velocity component at unstructured points
            v (array-like): V-velocity component at unstructured points (can be empty)
            w (array-like): W-velocity component at unstructured points (can be empty)
            vort (array-like): Vorticity values at unstructured points (can be empty)
            airfoil (bool): Whether to apply airfoil surface masking
        """
        self.calculate_grid(n, y_bnd, z_bnd, y, z, u, v, w, vort, airfoil)
        
    def calculate_grid(self, n: int, y_bnd: list, z_bnd: list, y, z, u, v, w, vort, airfoil: bool):
        """
        Perform the actual grid interpolation and optional airfoil masking.
        
        This method creates a structured rectangular grid and interpolates all flow
        variables from the unstructured input data. If airfoil masking is enabled,
        it identifies regions inside the airfoil surface and masks them appropriately.
        
        Args:
            n (int): Number of grid points in each direction
            y_bnd (list): Y-coordinate boundaries [y_min, y_max]
            z_bnd (list): Z-coordinate boundaries [z_min, z_max]
            y (array-like): Unstructured y-coordinates
            z (array-like): Unstructured z-coordinates
            u (array-like): U-velocity component
            v (array-like): V-velocity component
            w (array-like): W-velocity component
            vort (array-like): Vorticity values
            airfoil (bool): Enable airfoil surface masking
            
        Note:
            The airfoil surface is identified by regions where |u| < 1e-3,
            indicating near-zero velocity inside the solid body.
        """
        # Interpolating the data from unstructured grid to rectangular for plotting
        y_lin = np.linspace(min(y_bnd), max(y_bnd), num=n)
        z_lin = np.linspace(min(z_bnd), max(z_bnd), num=n)
        self.grid_y, self.grid_z = np.meshgrid(y_lin, z_lin)
        self.y, self.z = y, z
        self.u = u
        self.v = v
        self.w = w
        self.vort = vort
        # Interpolating
        self.grid_u = griddata(np.transpose([self.y, self.z]), self.u, (self.grid_y, self.grid_z), method='linear')
        self.grid_v = [] if len(v) == 0 else griddata(np.transpose([self.y, self.z]), self.v, (self.grid_y, self.grid_z), method='linear')
        self.grid_w = [] if len(w) == 0 else griddata(np.transpose([self.y, self.z]), self.w, (self.grid_y, self.grid_z), method='linear')
        self.grid_vort = [] if len(vort) == 0 else griddata(np.transpose([self.y, self.z]), self.vort, (self.grid_y, self.grid_z), method='linear')
        if airfoil:
            # Masking the values inside the airfoil surface as zero
            index_airf = np.where(np.abs(self.grid_u) < 1e-3)
            self.grid_u[index_airf] = float(0)
            if isinstance(self.grid_v, np.ndarray):
                self.grid_v[index_airf] = float(0)
            self.grid_w[index_airf] = float(0)
            self.grid_vort[index_airf] = float('nan')
            self.mask_y = [min(self.grid_y[index_airf]), max(self.grid_y[index_airf])]
            self.mask_z = [min(self.grid_z[index_airf]), max(self.grid_z[index_airf])]
            self.mask_indx = np.flip(np.isnan(self.grid_vort), axis=0)
