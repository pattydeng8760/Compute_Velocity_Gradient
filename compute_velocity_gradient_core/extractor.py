import numpy as np
import os
import h5py
from antares import Reader

def extract_gradient(arr, cut, reload:bool=False, output:str='./', time:int=None, data_type:str='LES'):
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
    Returns:
        _type_: _description_
    """
    assert data_type in ('LES', 'PIV'), f'The data type must be LES or RANS, but got {data_type}.'
    # The requried keys for the velocity gradient tensor
    VGT_keys = [
    ('grad_u_x', 'node'), ('grad_u_y', 'node'), ('grad_u_z', 'node'),
    ('grad_v_x', 'node'), ('grad_v_y', 'node'), ('grad_v_z', 'node'),
    ('grad_w_x', 'node'), ('grad_w_y', 'node'), ('grad_w_z', 'node')]

    print(f'----> Extracting velocity gradient tensor from h5 files with {data_type} Data.\n')
    print(arr[0])
    otuput_file_name = 'velocity_gradient_tensor_' + cut + '_' + data_type + '.h5'
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

        # Check the velocity gradient tensor components
        missing_keys = [key for key in VGT_keys if key not in b[0][0].keys()]
        if missing_keys:
            raise KeyError(f"The following required fields in the velocity gradient tensor are missing: {missing_keys}")
        else: 
            print('    All required fields in the velocity gradient tensor are present.')

        for idx, file in enumerate(arr):
            print(f'    Extracting data files from driectory {os.path.dirname(file)}')
            if idx % 100 == 0 or idx == 0 or idx == len(arr)-1:
                print(f'     Extracting file {idx+1}/{len(arr)}: {os.path.basename(file)}')
            r['filename'] = file
            b = r.read()
            # Extract ther velcoity vector
            if 'rho' not in b[0][0] or data_type == 'PIV':
                b[0][0]['rho'] = np.ones_like(b[0][0]['u'])
                b[0][0]['rho'] *= 1.225  # Constant density of air at sea level
            for i, key in enumerate([('u', 'node'), ('v', 'node'), ('w', 'node'),('rho', 'node'),('vort_x', 'node')]):
                velocity[i][:, idx] = b[0][0][key]
            # Extract the velocity gradient tensor
            for i, key in enumerate(VGT_keys):
                gradients[i][:, idx] = b[0][0][key]
        # Save the velocity gradient tensor
        node, time = gradients[0].shape
        velocity_gradient = np.array(gradients).reshape(3, 3, node, time)
        # Save the velocity and velocity gradient tensor as h5 file
        with h5py.File(os.path.join(output,otuput_file_name), 'w') as f:
            f.create_dataset('velocity', data=velocity, dtype='float32')
            f.create_dataset('velocity_gradient', data=velocity_gradient, dtype='float32')
    return velocity_gradient, velocity