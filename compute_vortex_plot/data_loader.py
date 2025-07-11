import os
import h5py
import numpy as np
from .utils import print

def load_velocity_invariants(location, data_type='LES'):
    """Load velocity invariants from HDF5 file."""
    file_path = os.path.join(os.getcwd(), 'Velocity_Invariants_' + location + '_' + data_type,
                             'Velocity_Invariants_' + location + '.h5')
    data = {}
    print("    Loading data from: {}".format(file_path))
    
    with h5py.File(file_path, 'r') as f:
        data['Phat_all'] = f['P'][:]
        data['Qhat_all'] = f['Q'][:]
        data['Rhat_all'] = f['R'][:]
        data['Qs_all'] = f['Qs'][:]
        data['Rs_all'] = f['Rs'][:]
        data['Qw_all'] = f['Qw'][:]
        data['x'] = f['x'][:]
        data['y'] = f['y'][:]
        data['z'] = f['z'][:]
        data['u'] = f['u'][:]
        data['v'] = f['v'][:]
        data['w'] = f['w'][:]
        data['vort_x'] = f['vort_x'][:]
        data['pressure_hessian'] = f['pressure_hessian'][:]
        data['var_A'] = f['Variance A'][:]
        data['var_S'] = f['Variance S'][:]
        data['var_omega'] = f['Variance Omega'][:]
        data['mean_SR'] = f['Mean strain_rate'][:]
    
    return data

def load_connectivity(location, data_type='LES'):
    """Load connectivity matrix from HDF5 file."""
    mesh_file = os.path.join(os.getcwd(), 'Velocity_Invariants_' + location + '_' + data_type,
                             'Velocity_Invariants_' + location + '_Mean.h5')
    with h5py.File(mesh_file, 'r') as f:
        connectivity = f['/0000/shared/connectivity/tri'][:]
    return connectivity