import os
import h5py
import numpy as np
from .utils import validate_file_exists, print

def load_velocity_invariants(cut, data_type='LES', base_dir=None):
    """Load velocity invariant data from HDF5 file."""
    if base_dir is None:
        base_dir = os.path.join(os.getcwd(), f'Velocity_Invariants_{cut}_{data_type}')
    
    file_path = os.path.join(base_dir, f'Velocity_Invariants_{cut}.h5')
    validate_file_exists(file_path, "Velocity invariants file")
    
    data = {}
    print(f"    Loading velocity invariant data from: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Load main invariant data
        data['Phat_all'] = f['P'][:]
        data['Qhat_all'] = f['Q'][:]
        data['Rhat_all'] = f['R'][:]
        data['Qs_all'] = f['Qs'][:]
        data['Rs_all'] = f['Rs'][:]
        data['Qw_all'] = f['Qw'][:]
        
        # Load coordinate data
        data['x'] = f['x'][:]
        data['y'] = f['y'][:]
        data['z'] = f['z'][:]
        
        # Load velocity data
        data['u'] = f['u'][:]
        data['v'] = f['v'][:]
        data['w'] = f['w'][:]
        
        # Load additional fields
        data['vort_x'] = f['vort_x'][:]
        data['pressure_hessian'] = f['pressure_hessian'][:]
        
        # Load variance data
        data['var_A'] = f['Variance A'][:]
        data['var_S'] = f['Variance S'][:]
        data['var_omega'] = f['Variance Omega'][:]
        data['mean_SR'] = f['Mean strain_rate'][:]
    
    print(f"    Loaded velocity invariant data with shape: {data['Phat_all'].shape}")
    return data

def load_connectivity(cut, data_type='LES', base_dir=None):
    """Load connectivity data from HDF5 file."""
    if base_dir is None:
        base_dir = os.path.join(os.getcwd(), f'Velocity_Invariants_{cut}_{data_type}')
    
    mesh_file = os.path.join(base_dir, f'Velocity_Invariants_{cut}_Mean.h5')
    validate_file_exists(mesh_file, "Connectivity mesh file")
    
    print(f"    Loading connectivity data from: {mesh_file}")
    
    with h5py.File(mesh_file, 'r') as f:
        connectivity = f['/0000/shared/connectivity/tri'][:]
    
    print(f"    Loaded connectivity with shape: {connectivity.shape}")
    return connectivity


def load_combined_core_data(data_file):
    """Load combined vortex core data from multiple locations."""
    validate_file_exists(data_file, "Combined core data file")
    
    print(f"    Loading combined core data from: {data_file}")
    
    combined_data = {}
    with h5py.File(data_file, 'r') as f:
        for location in f.keys():
            combined_data[location] = {}
            for vortex_type in f[location].keys():
                combined_data[location][vortex_type] = {}
                for variable in f[location][vortex_type].keys():
                    combined_data[location][vortex_type][variable] = f[location][vortex_type][variable][:]
    
    print(f"    Loaded combined data for locations: {list(combined_data.keys())}")
    return combined_data

def extract_qr_data(data_file, locations, vortex_type, num_features):
    """Extract Q and R data from combined core data file."""
    q, qs, qw, r, rs = [], [], [], [], []
    
    with h5py.File(data_file, 'r') as file:
        for loc in locations:
            if loc in file:
                loc_group = file[loc]
                if vortex_type in loc_group:
                    subgroup = loc_group[vortex_type]
                    if num_features is None:
                        num_features = file[loc][vortex_type]['Q'].shape[1]
                    
                    # Extract Q data
                    if 'Q' in subgroup:
                        q_row = subgroup['Q'][0]
                        q_row = q_row[:num_features] if q_row.shape[0] >= num_features else np.pad(q_row, (0, num_features - q_row.shape[0]))
                        q.append(q_row)
                    
                    # Extract Qs data
                    if 'Qs' in subgroup:
                        qs_row = subgroup['Qs'][0]
                        qs_row = qs_row[:num_features] if qs_row.shape[0] >= num_features else np.pad(qs_row, (0, num_features - qs_row.shape[0]))
                        qs.append(qs_row)
                    
                    # Extract Qw data
                    if 'Qw' in subgroup:
                        qw_row = subgroup['Qw'][0]
                        qw_row = qw_row[:num_features] if qw_row.shape[0] >= num_features else np.pad(qw_row, (0, num_features - qw_row.shape[0]))
                        qw.append(qw_row)
                    
                    # Extract R data
                    if 'R' in subgroup:
                        r_row = subgroup['R'][0]
                        r_row = r_row[:num_features] if r_row.shape[0] >= num_features else np.pad(r_row, (0, num_features - r_row.shape[0]))
                        r.append(r_row)
                    
                    # Extract Rs data
                    if 'Rs' in subgroup:
                        rs_row = subgroup['Rs'][0]
                        rs_row = rs_row[:num_features] if rs_row.shape[0] >= num_features else np.pad(rs_row, (0, num_features - rs_row.shape[0]))
                        rs.append(rs_row)
    
    return np.array(q), np.array(qs), np.array(qw), np.array(r), np.array(rs)