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


class PIVStructuredGrid:
    """
    PIV structured grid class that works directly with structured PIV data
    without interpolation.
    """
    
    def __init__(self, y_data, z_data, unique_y, unique_z, ny, nz):
        """
        Initialize PIV structured grid.
        
        Parameters:
        -----------
        y_data : array
            Y coordinates (1D array)
        z_data : array
            Z coordinates (1D array)
        unique_y : array
            Unique Y coordinate values
        unique_z : array
            Unique Z coordinate values
        ny : int
            Number of Y grid points
        nz : int
            Number of Z grid points
        """
        self.y_data = y_data
        self.z_data = z_data
        self.unique_y = unique_y
        self.unique_z = unique_z
        self.ny = ny
        self.nz = nz
        
        # Create meshgrid for plotting
        self.grid_y, self.grid_z = np.meshgrid(unique_y, unique_z, indexing='ij')
        
        # Store grid dimensions
        self.grid_shape = (ny, nz)
        
        # Initialize airfoil flag and mask
        self.airfoil = False
        self.mask_indx = None
        
        print(f"    PIV structured grid created: {ny} x {nz}")
        
    def add_variable(self, name, var_data):
        """
        Add a variable to the structured grid.
        
        Parameters:
        -----------
        name : str
            Variable name
        var_data : array
            Variable data (1D array matching grid points)
        """
        print(f"        Adding PIV variable: {name}")
        
        if len(var_data) == 0:
            # Handle empty variable
            setattr(self, name, np.array([]))
            return
        
        # Reshape 1D data to 2D structured grid
        try:
            # Try to reshape to structured grid
            var_2d = self._reshape_to_grid(var_data)
            setattr(self, name, var_2d)
            
            # Apply airfoil masking if this is velocity data
            if name == 'u' or name == 'vort_x':
                self._apply_airfoil_mask(name, var_2d)
                
        except Exception as e:
            print(f"        Warning: Could not reshape {name} to structured grid: {e}")
            # Fall back to 1D data
            setattr(self, name, var_data)
    
    def _reshape_to_grid(self, data):
        """Reshape 1D data to 2D structured grid."""
        if len(data) == self.ny * self.nz:
            # Perfect match - reshape directly
            return data.reshape(self.ny, self.nz)
        elif len(data) == len(self.y_data):
            # Data matches coordinate array size
            return self._interpolate_to_structured_grid(data)
        else:
            raise ValueError(f"Data length {len(data)} doesn't match grid structure")
    
    def _interpolate_to_structured_grid(self, data):
        """Interpolate irregular data to structured grid."""
        from scipy.interpolate import griddata
        
        # Create structured grid coordinates
        grid_y_flat = self.grid_y.flatten()
        grid_z_flat = self.grid_z.flatten()
        
        # Interpolate data to structured grid
        interpolated = griddata(
            np.column_stack((self.y_data, self.z_data)),
            data,
            np.column_stack((grid_y_flat, grid_z_flat)),
            method='linear',
            fill_value=np.nan
        )
        
        return interpolated.reshape(self.ny, self.nz)
    
    def _apply_airfoil_mask(self, var_name, var_data):
        """Apply airfoil masking to variables."""
        # For PIV data, identify airfoil region where velocity is very small
        if var_name == 'u':
            # Identify low-velocity regions (airfoil)
            mask_threshold = 1e-5  # More sensitive for PIV data
            airfoil_mask = np.abs(var_data) < mask_threshold
            
            # Only apply mask if significant airfoil region exists
            if np.sum(airfoil_mask) > 0.01 * var_data.size:
                self.airfoil = True
                self.index_airf = np.where(airfoil_mask)
                
                # Update the variable with masked values
                masked_var = var_data.copy()
                masked_var[self.index_airf] = 0.0
                setattr(self, var_name, masked_var)
                
                # Create mask index for plotting
                self.mask_indx = np.flip(airfoil_mask, axis=0)
                
                print(f"        Applied airfoil mask to {var_name}: {np.sum(airfoil_mask)} points masked")
            else:
                self.airfoil = False
        
        elif var_name == 'vort_x' and hasattr(self, 'airfoil') and self.airfoil:
            # Apply existing airfoil mask to vorticity
            masked_var = var_data.copy()
            masked_var[self.index_airf] = np.nan
            setattr(self, var_name, masked_var)
    
    def calculate_grid(self, n, y_bnd, z_bnd, var, name):
        """
        Compatibility method for LES interface.
        For PIV structured grid, this just adds the variable directly.
        """
        self.add_variable(name, var)