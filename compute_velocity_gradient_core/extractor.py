import numpy as np
import os
import h5py
from antares import Reader

def extract_gradient(arr, cut, reload:bool=False, output:str='./', time:int=None):
    """
    Extract the velocity gradient tensor from the h5 files.
    NOTE: The .h5 files are extracted from the extract_cutplane module and must contain the velocity gradient tensor components as grad_u_x style variable names
    Args:
        arr (list): The list of h5 files to extract the velocity gradient tensor from. Post extraction of the cutplanes
        cut (string): The cut name to extract the velocity gradient tensor from. This is used to name the output file. This should be the same as the cut name used in the extract_cutplane module.
        reload (bool, optional): If True, the velocity gradient tensor will be reloaded from the h5 file. If False, the velocity gradient tensor will be extracted from the h5 files. Defaults to False.
        output (str, optional): The output directory to save the velocity gradient tensor. Defaults to './'.
        time (int, optional): The end time step to extract the velocity gradient tensor from. If None, all time steps will be extracted. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    if os.path.exists(output+'/velocity_gradient_tensor_' + cut + '.h5') and not reload:
        print('----> VGT already extracted.')
        print(f'Reloading extracrted velocity gradient tensor from {output}/velocity_gradient_tensor_{cut}.h5')
        with h5py.File(output+'/velocity_gradient_tensor_' + cut + '.h5', 'r') as f:
            velocity = f['velocity'][:]
            velocity_gradient = f['velocity_gradient'][:]
        if time is not None:
            velocity_gradient = velocity_gradient[:, :, :, :time]
            velocity = [v[:, :time] for v in velocity]
            print(f'---->Extracted velocity gradient tensor from h5 files with {time} time steps and {velocity_gradient.shape[2]} nodes.')
    else:
        print('----> Extracting velocity gradient tensor from h5 files.\n')
        r = Reader('hdf_antares')
        r['filename'] = arr[0]
        b = r.read()  # Base object of the Antares API
        velocity = [np.zeros((np.shape(b[0][0]['u'])[0], len(arr)), dtype='float32') for _ in range(5)]
        gradients = [np.zeros((np.shape(b[0][0]['grad_u_x'])[0], len(arr)), dtype='float32') for _ in range(9)]
        for idx, file in enumerate(arr):
            if idx == 0 or idx == len(arr)-1:
                print(f'Extracting file {file}')
            elif idx % 100 == 0:
                print(f'Extracting file {os.path.basename(file)}')
            r['filename'] = file
            b = r.read()
            # Eextract ther velcoity vector
            for i, key in enumerate([('u', 'node'), ('v', 'node'), ('w', 'node'),('rho', 'node'),('vort_x', 'node')]):
                velocity[i][:, idx] = b[0][0][key]
            # Extract the velocity gradient tensor
            for i, key in enumerate([('grad_u_x', 'node'), ('grad_u_y', 'node'), ('grad_u_z', 'node'),
                                     ('grad_v_x', 'node'), ('grad_v_y', 'node'), ('grad_v_z', 'node'),
                                     ('grad_w_x', 'node'), ('grad_w_y', 'node'), ('grad_w_z', 'node')]):
                gradients[i][:, idx] = b[0][0][key]
        # Save the velocity gradient tensor
        node, time = gradients[0].shape
        velocity_gradient = np.array(gradients).reshape(3, 3, node, time)
        # Save the velocity and velocity gradient tensor as h5 file
        with h5py.File(output+'/velocity_gradient_tensor_' + cut + '.h5', 'w') as f:
            f.create_dataset('velocity', data=velocity, dtype='float32')
            f.create_dataset('velocity_gradient', data=velocity_gradient, dtype='float32')
    return velocity_gradient, velocity