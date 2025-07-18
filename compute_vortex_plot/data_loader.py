import os
import h5py
import numpy as np
from .utils import print

def load_velocity_invariants(location, data_type='LES', limited_gradient=False):
    """Load velocity invariants from HDF5 file."""
    parent_dir = 'Velocity_Invariants_' + location + '_' + data_type
    file_name = 'Velocity_Invariants_' + location 
    if limited_gradient:
        parent_dir += '_Limited'
        file_name += '_Limited'
    file_path = os.path.join(os.getcwd(), parent_dir, file_name + '.h5')
    data = {}
    print("    Loading data from: {}".format(file_path))
    
    with h5py.File(file_path, 'r') as f:
        data['Phat_all'] = f['P'][:]
        data['Qhat_all'] = f['Q'][:]
        data['Rhat_all'] = f['R'][:]
        data['Qs_all'] = f['Qs'][:]
        data['Rs_all'] = f['Rs'][:]
        data['Qw_all'] = f['Qw'][:]
        data['u'] = f['u'][:]
        data['v'] = f['v'][:]
        data['w'] = f['w'][:]
        data['vort_x'] = f['vort_x'][:]
        data['pressure_hessian'] = f['pressure_hessian'][:]
        data['var_A'] = f['Variance A'][:]
        data['var_S'] = f['Variance S'][:]
        data['var_omega'] = f['Variance Omega'][:]
        data['mean_SR'] = f['Mean strain_rate'][:]
        
        # Handle coordinate system differences between LES and PIV
        if data_type == 'PIV':
            # PIV coordinate system: x=wall-normal, y=span, z=streamwise
            # Map to LES coordinate system: x=streamwise, y=wall-normal, z=span
            data['x'] = f['z'][:]  # PIV z (streamwise) -> LES x (streamwise)
            data['y'] = f['x'][:]  # PIV x (wall-normal) -> LES y (wall-normal)
            data['z'] = f['y'][:]  # PIV y (span) -> LES z (span)
            print(f"    PIV coordinate system mapped to LES system")
            print(f"    Data shape: {data['y'].shape} points")
        else:
            # LES coordinate system: x=streamwise, y=wall-normal, z=span
            data['x'] = f['x'][:]
            data['y'] = f['y'][:]
            data['z'] = f['z'][:]
            print(f"    LES coordinate system loaded")
            print(f"    Data shape: {data['y'].shape} points")
    
    return data

def load_connectivity(location, data_type='LES', limited_gradient=False):
    """Load connectivity matrix from HDF5 file."""
    parent_dir = 'Velocity_Invariants_' + location + '_' + data_type
    file_name = 'Velocity_Invariants_' + location + '_Mean'
    if limited_gradient:
        parent_dir += '_Limited'
        file_name += '_Limited'
    mesh_file = os.path.join(os.getcwd(), parent_dir, file_name + '.h5')
    with h5py.File(mesh_file, 'r') as f:
        if data_type == 'PIV':
            # PIV data uses quadrilateral connectivity stored in instants/0000/connectivity/qua
            connectivity = f['/0000/instants/0000/connectivity/qua'][:]
            print(f"    PIV connectivity loaded: {connectivity.shape} quadrilaterals")
        else:
            # LES data uses triangular connectivity stored in shared/connectivity/tri
            connectivity = f['/0000/shared/connectivity/tri'][:]
            print(f"    LES connectivity loaded: {connectivity.shape} triangles")
    return connectivity