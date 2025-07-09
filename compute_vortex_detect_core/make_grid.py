
import numpy as np

class make_grid:
    """
    This is a class to create a grid for plotting the velocity and vorticity fields.
    It takes the number of grid points, boundaries for y and z, and the velocity and vorticity data.
    The grid is created using linear interpolation from the unstructured data to a structured grid for LES data. 
    """
    
    def __init__(self, n: int, y_bnd: list, z_bnd: list, y, z, u, v, w, vort, airfoil: bool):
        self.calculate_grid(n, y_bnd, z_bnd, y, z, u, v, w, vort, airfoil)
        
    def calculate_grid(self, n: int, y_bnd: list, z_bnd: list, y, z, u, v, w, vort, airfoil: bool):
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
