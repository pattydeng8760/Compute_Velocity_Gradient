import numpy as np
import os
import h5py
from antares import Reader
from .utils import print

def extract_gradient(arr, cut, reload:bool=False, output:str='./', time:int=None, data_type:str='LES', limited_gradient:bool=False):
    """
    Extract the velocity gradient tensor from the h5 files.
    NOTE: The .h5 files are extracted from the extract_cutplane module and must contain the velocity gradient tensor components as grad_u_x style variable names
    Args:
        arr (list): The list of h5 files to extract the velocity gradient tensor from. Post extraction of the cutplanes
        cut (string): The cut name to extract the velocity gradient tensor from. This is used to name the output file. This should be the same as the cut name used in the extract_cutplane module.
        reload (bool, optional): If True, the velocity gradient tensor will be reloaded from the h5 file. If False, the velocity gradient tensor will be extracted from the h5 files. Defaults to False.
        output (str, optional): The output directory to save the velocity gradient tensor. Defaults to './'.
        time (int, optional): The end time step to extract the velocity gradient tensor from. If None, all time steps will be extracted. Defaults to None.
        data_type (str, optional): The type of data to extract the velocity gradient tensor from. Defaults to 'LES'. Can be 'LES' or 'PIV' depending on the underlying data source.
        limited_gradient (bool, optional): If True and data_type is 'LES', computes limited gradient tensor corresponding to stereo-PIV availability where dv/dx and dw/dx are unavailable, and du/dx is calculated from incompressible assumption. Defaults to False.
    Returns:
        _type_: _description_
    """
    assert data_type in ('LES', 'PIV'), f'The data type must be LES or RANS, but got {data_type}.'
    # The requried keys for the velocity gradient tensor
    VGT_keys = [
    ('grad_u_x', 'node'), ('grad_u_y', 'node'), ('grad_u_z', 'node'),
    ('grad_v_x', 'node'), ('grad_v_y', 'node'), ('grad_v_z', 'node'),
    ('grad_w_x', 'node'), ('grad_w_y', 'node'), ('grad_w_z', 'node')]

    # Modify output filename if limited gradient is enabled
    gradient_type = "limited" if limited_gradient and data_type == 'LES' else "full"
    print(f'----> Extracting velocity gradient tensor from h5 files with {data_type} Data ({gradient_type} gradient).')
    
    output_suffix = data_type
    if limited_gradient and data_type == 'LES':
        output_suffix += '_Limited'
    otuput_file_name = 'velocity_gradient_tensor_' + cut + '_' + output_suffix + '.h5'
    if os.path.exists(os.path.join(output,otuput_file_name)) and not reload:
        print('----> VGT already extracted.')
        print(f'     Reloading extracrted velocity gradient tensor from {output}/velocity_gradient_tensor_{cut}.h5')
        with h5py.File(os.path.join(output,otuput_file_name), 'r') as f:
            velocity = f['velocity'][:]
            velocity_gradient = f['velocity_gradient'][:]
        if time is not None:
            velocity_gradient = velocity_gradient[:, :, :, :time]
            velocity = [v[:, :time] for v in velocity]
            print(f'---->Extracted velocity gradient tensor from h5 files with {time} time steps and {velocity_gradient.shape[2]} nodes.')
    else:
        r = Reader('hdf_antares')
        r['filename'] = arr[0]
        b = r.read()  # Base object of the Antares API
        # Initialize the velocity and velocity gradient tensor
        velocity = [np.zeros((np.shape(b[0][0]['u'])[0], len(arr)), dtype='float32') for _ in range(5)]
        gradients = [np.zeros((np.shape(b[0][0]['grad_u_x'])[0], len(arr)), dtype='float32') for _ in range(9)]
        print("    The shape of the velocity vector is (nodes, time): ", velocity[0].shape)
        print("    The shape of the velocity gradient tensor is (nodes, time): ", gradients[0].shape)
        # Check the velocity gradient tensor components
        missing_keys = [key for key in VGT_keys if key not in b[0][0].keys()]
        if missing_keys:
            raise KeyError(f"The following required fields in the velocity gradient tensor are missing: {missing_keys}")
        else: 
            print('    All required fields in the velocity gradient tensor are present.')
        print(f'\nExtracting data files from driectory {os.path.dirname(arr[0])}...')
        for idx, file in enumerate(arr):
            if idx % 100 == 0 or idx == 0 or idx == len(arr)-1:
                print(f'     Extracting file {idx}/{len(arr)}: {os.path.basename(file)}')
            r['filename'] = file
            b = r.read()
            # Extract ther velcoity vector
            if 'rho' not in b[0][0] or data_type == 'PIV':
                b[0][0]['rho'] = np.ones_like(b[0][0]['u'])
                b[0][0]['rho'] *= 1.225  # Constant density of air at sea level
            if 'pressure' not in b[0][0] or data_type == 'PIV':
                b[0][0]['pressure'] = np.ones_like(b[0][0]['u'])
                b[0][0]['pressure'] *= 101325
            for i, key in enumerate([('u', 'node'), ('v', 'node'), ('w', 'node'),('rho', 'node'),('vort_x', 'node'),('pressure', 'node')]):
                velocity[i][:, idx] = b[0][0][key]
            # Extract the velocity gradient tensor
            for i, key in enumerate(VGT_keys):
                gradients[i][:, idx] = b[0][0][key]
        
        # Apply limited gradient modification for LES data if requested
        if (limited_gradient and data_type == 'LES') or data_type == 'PIV':
            print('    Applying limited gradient computation for stereo-PIV compatibility...')
            # Set dv/dx (index 3) and dw/dx (index 6) to zero
            gradients[3][:, :] = 0.0  # grad_v_x = dv/dx = 0
            gradients[6][:, :] = 0.0  # grad_w_x = dw/dx = 0
            
            # Calculate du/dx from incompressible assumption: du/dx = -(dv/dy + dw/dz)
            # gradients[0] = grad_u_x, gradients[4] = grad_v_y, gradients[8] = grad_w_z
            gradients[0][:, :] = -(gradients[4][:, :] + gradients[8][:, :])  # du/dx = -(dv/dy + dw/dz)
            print('    Limited gradient modifications applied:')
            print('      - dv/dx set to zero')
            print('      - dw/dx set to zero')
            print('      - du/dx calculated from incompressible assumption: du/dx = -(dv/dy + dw/dz)')
        
        # Save the velocity gradient tensor
        node, time = gradients[0].shape
        velocity_gradient = np.array(gradients).reshape(3, 3, node, time)
        # Save the velocity and velocity gradient tensor as h5 file
        with h5py.File(os.path.join(output,otuput_file_name), 'w') as f:
            f.create_dataset('velocity', data=velocity, dtype='float32')
            f.create_dataset('velocity_gradient', data=velocity_gradient, dtype='float32')
    return velocity_gradient, velocity