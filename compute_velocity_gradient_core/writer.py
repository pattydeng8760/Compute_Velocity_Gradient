from .utils import print
import os
import h5py
import numpy as np
from antares import Reader, Writer


def save_output_main(node,node_indices,time,results,arr,output,velocity,cut,limited_gradient=False):
    # Initialize arrays for final results
    P_final, Q_final, R_final = [np.zeros([node, time],dtype='float32') for _ in range(3)]
    strain_rate_final, pressure_hessian_final = [np.zeros([node, time],dtype='float32') for _ in range(2)]
    
    # Reassemble the global result array
    for part_P, part_Q, part_R, strain_rate, pressure_hessian, block_num in results:
        indices = node_indices[block_num]
        P_final[indices, :], Q_final[indices, :], R_final[indices, :] = part_P, part_Q, part_R
        strain_rate_final[indices, :] = strain_rate
        pressure_hessian_final[indices, :] = pressure_hessian
    del results         # Free up memory
    print('\n----> Saving results...')
    #print('Saving mean PQR')
    # Saving the output mean PQR for visualization
    filename = 'Velocity_Invariants_' + cut + '_Mean'
    if limited_gradient:
        filename += '_Limited'
    filename = os.path.join(output,filename)
    r = Reader('hdf_antares')
    r['filename'] = arr[0]          # use the first file to get the base object and overwrite the file contents 
    b = r.read()  # b is the Base object of the Antares API
    b[0][0]['P'] = np.mean(P_final, axis=1)
    b[0][0]['Q'] = np.mean(Q_final, axis=1)
    b[0][0]['R'] = np.mean(R_final, axis=1)
    b[0][0]['vort_x'] = np.mean(velocity[4], axis=1)
    b[0][0]['strain_rate'] = np.mean(strain_rate_final, axis=1)
    b[0][0]['Mean pressure_hessian'] = np.mean(pressure_hessian_final, axis=1)
    b[0][0]['Variance pressure_hessian'] = np.std(pressure_hessian_final, axis=1)
    b[0][0]['u'] = np.mean(velocity[0], axis=1)
    b[0][0]['v'] = np.mean(velocity[1], axis=1)
    b[0][0]['w'] = np.mean(velocity[2], axis=1)
    b[0][0]['density'] = np.mean(velocity[3], axis=1)
    w = Writer('hdf_antares')  # This is another format (still hdf5)
    w['base'] = b[:, :, ['x', 'y', 'z', 'P', 'Q', 'R', 'vort_x', 'strain_rate', 'u', 'v', 'w','Mean pressure_hessian','Variance pressure_hessian','density']]
    w['filename'] = filename
    w.dump()
    print('    Mean PQR saved to {0:s}.'.format(filename))
    
    #print('Saving Full PQR')
    # Saving the full VGT result as h5 file
    filename = 'Velocity_Invariants_' + cut
    if limited_gradient:
        filename += '_Limited'
    output_file = os.path.join(output, filename + '.h5')
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('P', data=P_final, dtype='float32')
        f.create_dataset('Q', data=Q_final, dtype='float32')
        f.create_dataset('R', data=R_final, dtype='float32')
        f.create_dataset('x', data=b[0][0]['x'], dtype='float32')
        f.create_dataset('y', data=b[0][0]['y'], dtype='float32')
        f.create_dataset('z', data=b[0][0]['z'], dtype='float32')
        f.create_dataset('vort_x', data=velocity[4], dtype='float32')
        f.create_dataset('strain_rate', data=strain_rate_final, dtype='float32')
        f.create_dataset('pressure_hessian', data=pressure_hessian_final, dtype='float32')
        f.create_dataset('u', data=velocity[0], dtype='float32')
        f.create_dataset('v', data=velocity[1], dtype='float32')
        f.create_dataset('w', data=velocity[2], dtype='float32')
    print('    Full PQR saved to {0:s}.'.format(output_file))
    #print('\n---->File saving complete.')

def save_output_strain(node,node_indices,time,results,arr,output,velocity,cut,limited_gradient=False):
    # Initialize arrays for final results
    Qs,Rs,Qw = [np.zeros([node, time],dtype='float32') for _ in range(3)]
    var_A, var_S, var_Omega = [np.zeros([node],dtype='float32') for _ in range(3)]
    strain_rate_mean, rotation_rate_mean, strain_rate_fluc_rms, rotation_rate_fluc_rms = [np.zeros([node],dtype='float32') for _ in range(4)]
    # Reassemble the global result array
    for Q_S_part, R_S_part, Q_W_part, var_A_part, var_S_part, var_Omega_part, strain_rate_mean_part, rotation_rate_mean_part, strain_rate_rms_part, rotation_rate_rms_part, block_num in results:
        indices = node_indices[block_num]
        Qs[indices, :],Rs[indices, :], Qw[indices, :] = Q_S_part, R_S_part, Q_W_part
        var_A[indices], var_S[indices], var_Omega[indices],  = var_A_part, var_S_part, var_Omega_part
        strain_rate_mean[indices], rotation_rate_mean[indices] = strain_rate_mean_part, rotation_rate_mean_part
        strain_rate_fluc_rms[indices], rotation_rate_fluc_rms[indices] = strain_rate_rms_part, rotation_rate_rms_part
    del results         # Free up memory
    print('\n---->Saving results...')
    #print('Saving mean Qs, Rs, Qw')
    # Saving the output mean PQR for visualization
    filename = 'Velocity_Invariants_Rotation_Strain_' + cut + '_Mean'
    if limited_gradient:
        filename += '_Limited'
    filename = os.path.join(output,filename)
    r = Reader('hdf_antares')
    r['filename'] = arr[0]
    b = r.read()  # b is the Base object of the Antares API
    b[0][0]['Qs'] = np.mean(Qs, axis=1)
    b[0][0]['Variance Qs'] = np.std(Qs, axis=1)
    b[0][0]['Rs'] = np.mean(Rs, axis=1)
    b[0][0]['Variance Rs'] = np.std(Rs, axis=1)
    b[0][0]['Qw'] = np.mean(Qw, axis=1)
    b[0][0]['Variance Qw'] = np.std(Qw, axis=1)
    b[0][0]['Variance A'] = var_A
    b[0][0]['Variance S'] = var_S
    b[0][0]['Variance Omega'] = var_Omega
    b[0][0]['Mean strain_rate'] = strain_rate_mean
    b[0][0]['Mean rotation_rate'] = rotation_rate_mean
    b[0][0]['Mean strain_rate_fluc_rms'] = strain_rate_fluc_rms
    b[0][0]['Mean rotation_rate_fluc_rms'] = rotation_rate_fluc_rms
    w = Writer('hdf_antares')  # This is another format (still hdf5)
    w['base'] = b[:, :, ['x', 'y', 'z', 'Qs', 'Rs', 'Qw', 'Variance Qs', 'Variance Rs', 'Variance Qw','Variance A','Variance S','Variance Omega','Mean strain_rate']]
    w['filename'] = filename
    w.dump()
    print('    Mean Qs, Rs, Qw saved to {0:s}.'.format(filename))
    
    #print('Saving Full Qs, Rs, Qw')
    # Saving the full VGT result as h5 file
    filename = 'Velocity_Invariants_' + cut + '.h5'
    if limited_gradient:
        filename += '_Limited'
    output_file = os.path.join(output, filename)
    with h5py.File(output_file, 'a') as f:
        f.create_dataset('Qs', data=Qs, dtype='float32')
        f.create_dataset('Rs', data=Rs, dtype='float32')
        f.create_dataset('Qw', data=Qw, dtype='float32')
        f.create_dataset('Variance A', data=var_A, dtype='float32')
        f.create_dataset('Variance S', data=var_S, dtype='float32')
        f.create_dataset('Variance Omega', data=var_Omega, dtype='float32')
        f.create_dataset('Mean strain_rate', data=strain_rate_mean, dtype='float32')
        f.create_dataset('Mean rotation_rate', data=rotation_rate_mean, dtype='float32')
        f.create_dataset('Mean strain_rate_fluc_rms', data=strain_rate_fluc_rms, dtype='float32')
        f.create_dataset('Mean rotation_rate_fluc_rms', data=rotation_rate_fluc_rms, dtype='float32')
    print('    Full Qs, Rs, Qw appended to {0:s}.'.format(output_file))
    #print('\n---->File saving complete.')