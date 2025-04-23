from antares import Reader, Writer
import numpy as np
import os
import glob
import h5py
import logging
import builtins
import sys
from sys import argv
from multiprocessing import Pool, cpu_count, Value, Lock
import time

# Setup basic configuration for logging
sys.stdout = open(os.path.join('log_invariants_'+str(argv[1])+'.txt'), "w", buffering=1)
# Shared counter initialization
counter = Value('i', 0)
counter_lock = Lock()
# Printing any on-screen print functions into a log file
def print(text:str,**kwargs):
    """ print function to print to the screen and to a log file
    """
    builtins.print(text,**kwargs)
    os.fsync(sys.stdout)

def timer(func):
    """ Decorator to time the function func to track the time taken for the function to run"""
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print('The total compute time is: {0:1.0f} s'.format(elapsed))
        return elapsed
    return inner

def increment_counter():
    """Safely increments the shared counter and returns its value."""
    with counter_lock:
        counter.value += 1
        return counter.value

def compute_PQR_mod(images_part, block_num):
    """Main worker function to compute the velocity gradient invariants at each node position
    the PQR are invariants of the velocity gradient tensor A_ij = du_i/dx_j
    the invariants are normalized by the temporal variance of the velocity gradient tensor over the time interval
    Args:
        images_part: The velocity gradient tensor at each node position of the partitioned data with size (3, 3, node_count, time_int)
        block_num: The block number of the partitioned data

    Returns:
        P_hat_part: The normalized velocity gradient invariant P_hat at each node position
        Q_hat_part: The normalized velocity gradient invariant Q_hat at each node position
        R_hat_part: The normalized velocity gradient invariant R_hat at each node position
        vort_x_part: The x-component of the vorticity vector at each node position
        strain_rate_part: The strain rate magnitude at each node position
        block_num: The block number of the partitioned data
    """
    node_count, time_int = images_part.shape[2], images_part.shape[3]
    
    # Initialize output arrays
    P_hat_part = np.zeros((node_count, time_int))
    Q_hat_part = np.zeros((node_count, time_int))
    R_hat_part = np.zeros((node_count, time_int))
    strain_rate_part = np.zeros((node_count, time_int))
    pressure_hessian_part = np.zeros((node_count, time_int))
    
    if np.mod(block_num+1, 100) == 0 or block_num == 0:
        print(f'Processing iteration block number {block_num}')
    
    # Compute velocity gradient invariants for each node
    for node_idx in range(node_count):
        vel_grad = images_part[:, :, node_idx, :]
        mean_A = np.mean(vel_grad, axis=2, keepdims=True)
        
        # Compute fluctuating velocity gradient tensor
        A_fluc = vel_grad - mean_A
        
        # Compute variance tensor
        var_A_time = np.einsum('ijT,ikT->jkT', A_fluc, A_fluc)
        
        # Compute the temporal variance for normalization
        var_A = np.mean(var_A_time[0, 0, :] + var_A_time[1, 1, :] + var_A_time[2, 2, :])
        
        # Compute invariants
        tr_A = np.trace(vel_grad, axis1=0, axis2=1)
        P_hat_part[node_idx, :] = -tr_A
        Q_hat_part[node_idx, :] = 0.5 * (tr_A**2 - np.trace(np.einsum('ijt,jkt->ikt', vel_grad, vel_grad), axis1=0, axis2=1)) / var_A
        R_hat_part[node_idx, :] = -np.linalg.det(vel_grad.transpose(2, 0, 1)) / var_A**1.5
        
        # Compute strain rate magnitude
        #S = 0.5 * (vel_grad + vel_grad.transpose(1, 0, 2))
        #strain_rate_part[node_idx, :] = np.sqrt(2 * np.sum(S * S, axis=(0, 1)))
        strain_rate_part[node_idx, :] = np.sqrt(2 * np.sum((0.5 * (vel_grad + vel_grad.transpose(1, 0, 2)) * 0.5 * (vel_grad + vel_grad.transpose(1, 0, 2))), axis=(0, 1))) 
        #omega_x = vel_grad[2, 1, :] - vel_grad[1, 2, :]
        #omega_y = vel_grad[0, 2, :] - vel_grad[2, 0, :]
        #omega_z = vel_grad[1, 0, :] - vel_grad[0, 1, :]
        # Compute pressure Hessian magnitude
        #enstrophy = 0.5 * ((vel_grad[2, 1, :] - vel_grad[1, 2, :])**2 + (vel_grad[0, 2, :] - vel_grad[2, 0, :])**2 + (vel_grad[1, 0, :] - vel_grad[0, 1, :])**2)
        #strain = np.sum(A_fluc[:3, :3, :]**2, axis=(0, 1)) + 0.5 * np.sum((A_fluc + A_fluc.transpose(1, 0, 2))**2, axis=(0, 1))
        #pressure_hessian_part[node_idx, :] = enstrophy - strain
        pressure_hessian_part[node_idx,:] = 0.5 * ((vel_grad[2, 1, :] - vel_grad[1, 2, :])**2 + (vel_grad[0, 2, :] - vel_grad[2, 0, :])**2 + (vel_grad[1, 0, :] - vel_grad[0, 1, :])**2) \
            -  np.sum(A_fluc[:3, :3, :]**2, axis=(0, 1)) + 0.5 * np.sum((A_fluc + A_fluc.transpose(1, 0, 2))**2, axis=(0, 1))
        
    return P_hat_part, Q_hat_part, R_hat_part, strain_rate_part ,pressure_hessian_part, block_num


def compute_PQR_vectorized(images_part, block_num):
    """
    Compute velocity gradient invariants (P, Q, R), vorticity, strain rate, 
    and pressure hessian vector in a fully vectorized manner.
    
    Args:
        images_part: np.ndarray of shape (3, 3, node_count, time_int)
        block_num: block number of the partitioned data
        
    Returns:
        P_hat:   Normalized invariant P_hat, shape (node_count, time_int)
        Q_hat:   Normalized invariant Q_hat, shape (node_count, time_int)
        R_hat:   Normalized invariant R_hat, shape (node_count, time_int)
        vort:    Vorticity magnitude, shape (node_count, time_int)
        vort_x:  x-component of vorticity, shape (node_count, time_int)
        strain_rate: Strain rate magnitude, shape (node_count, time_int)
        pressure_hessian: Pressure hessian magnitude, shape (node_count, time_int)
        block_num: The provided block number
    """
    # Extract dimensions
    _, _, node_count, time_int = images_part.shape

    # --- Compute mean and fluctuations over time ---
    # Mean of the velocity gradient tensor over time (for each node)
    # Resulting shape: (3, 3, node_count)
    mean_A = np.mean(images_part, axis=3)
    
    # Fluctuating part: A_fluc = A - <A>, broadcast mean_A to time dimension
    # Shape: (3, 3, node_count, time_int)
    A_fluc = images_part - mean_A[..., None]

    # --- Compute var_A (used for normalization) ---
    # For each node and time, the Frobenius norm squared is the sum of squares of the fluctuation.
    # Sum over the first two axes (the 3x3 matrices) gives shape (node_count, time_int),
    # then average over time (axis=1) gives var_A of shape (node_count,).
    A_fluc_sq = np.sum(A_fluc**2, axis=(0, 1))  # shape: (node_count, time_int)
    var_A = np.mean(A_fluc_sq, axis=1)  # shape: (node_count,)

    # --- Compute P_hat ---
    # Trace of A: sum of diagonal elements. Since images_part has shape (3,3,node_count,time_int),
    # each diagonal (e.g. images_part[0,0,:,:]) is of shape (node_count, time_int).
    trace_A = images_part[0, 0, :, :] + images_part[1, 1, :, :] + images_part[2, 2, :, :]
    # Invariant P_hat is minus the trace.
    P_hat = -trace_A  # shape: (node_count, time_int)

    # --- Compute Q_hat ---
    # Q_hat = 0.5 * (trace(A)**2 - trace(A @ A)) / var_A
    # To compute trace(A @ A), note that:
    #   trace(A @ A) = sum_ij A[i,j]*A[j,i].
    # We can compute this by taking elementwise product of images_part and its transpose (swapaxes 0 and 1).
    trace_AA = np.sum(images_part * images_part.swapaxes(0, 1), axis=(0, 1))  # shape: (node_count, time_int)
    # Use broadcasting for var_A (shape (node_count,) -> (node_count, time_int))
    Q_hat = 0.5 * (trace_A**2 - trace_AA) / var_A[:, None]

    # --- Compute R_hat ---
    # R_hat = -det(A)/var_A^(1.5)
    # To compute determinants vectorized, first reshape so that the last two axes are the 3x3 matrices.
    # Here we simply transpose the array so that shape becomes (node_count, time_int, 3, 3)
    A_for_det = images_part.transpose(2, 3, 0, 1)
    det_A = np.linalg.det(A_for_det)  # shape: (node_count, time_int)
    R_hat = -det_A / (var_A[:, None]**1.5)

    # --- Compute strain rate magnitude ---
    # Strain rate tensor S = 0.5*(A + A.T)
    # The transpose here swaps axes 0 and 1.
    S = 0.5 * (images_part + images_part.swapaxes(0, 1))
    # Sum of squares of S (elementwise) over the matrix indices gives shape (node_count, time_int)
    strain_rate = np.sqrt(2 * np.sum(S**2, axis=(0, 1)))

    # --- Compute pressure hessian magnitude ---
    # First, compute quantities from the fluctuating part A_fluc.
    omega_fluc_x = A_fluc[2, 1, :, :] - A_fluc[1, 2, :, :]
    omega_fluc_y = A_fluc[0, 2, :, :] - A_fluc[2, 0, :, :]
    omega_fluc_z = A_fluc[1, 0, :, :] - A_fluc[0, 1, :, :]
    enstrophy = 0.5 * (omega_fluc_x**2 + omega_fluc_y**2 + omega_fluc_z**2)
    
    strain_diag = (A_fluc[0, 0, :, :]**2 + 
                   A_fluc[1, 1, :, :]**2 + 
                   A_fluc[2, 2, :, :]**2)
    strain_offdiag = ((A_fluc[0, 1, :, :] + A_fluc[1, 0, :, :])**2 +
                      (A_fluc[0, 2, :, :] + A_fluc[2, 0, :, :])**2 +
                      (A_fluc[1, 2, :, :] + A_fluc[2, 1, :, :])**2)
    strain = strain_diag + 0.5 * strain_offdiag
    pressure_hessian = enstrophy - strain
    if np.mod(block_num+1, 100) == 0 or block_num == 0:
        print(f'    Processing iteration block number {block_num+1}')
    return P_hat, Q_hat, R_hat, strain_rate, pressure_hessian, block_num


def compute_SQW_invariants(images_part, block_num):
    """
    Computes the invariants of the rate-of-strain tensor (S_ij) and rate-of-rotation tensor (Ω_ij)
    for each node position in the partitioned data.

    Args:
        images_part: Velocity gradient tensor A_ij = du_i/dx_j of shape (3, 3, node_count, time_int)
        block_num: Block number of the partitioned data

    Returns:
        Q_S_part: Second invariant of the rate-of-strain tensor Q_S
        R_S_part: Third invariant of the rate-of-strain tensor R_S
        Q_W_part: Invariant of the rate-of-rotation tensor Q_W
        block_num: Block number of the partitioned data
    """
    node_count, time_int = images_part.shape[2], images_part.shape[3]

    # Initialize output arrays
    Q_S_part = np.zeros((node_count, time_int))
    R_S_part = np.zeros((node_count, time_int))
    Q_W_part = np.zeros((node_count, time_int))
    
    if np.mod(block_num+1, 100) == 0 or block_num == 0:
        print(f'    Processing iteration block number {block_num+1}')
        
    # Compute invariants at each node
    for node_idx in range(node_count):
        A = images_part[:, :, node_idx, :]  # Shape: (3, 3, time_int)
        A = A - np.mean(A, axis=2, keepdims=True)  # Subtract the mean velocity gradient tensor to obtain fluctuating velocity gradient tensor
        
        # Compute strain-rate tensor S_ij and rotation tensor Ω_ij
        S = 0.5 * (A + A.transpose(1, 0, 2))  # Symmetric part
        # The variable S for normalization
        var_S_time = np.einsum('ijT,ikT->jkT', S,S)
        # Compute the temporal variance for normalization
        var_S = np.mean(var_S_time[0, 0, :] + var_S_time[1, 1, :] + var_S_time[2, 2, :])
        
        OMEGA = 0.5 * (A - A.transpose(1, 0, 2))  # Antisymmetric part
        
        # Compute Q_S = -0.5 * S_ij S_ij
        Q_S_part[node_idx, :] = -0.5 * np.einsum('ijt,ijt->t', S, S)

        # Compute R_S = -1/3 * S_ij S_jk S_ki
        R_S_part[node_idx, :] = -1/3 * np.einsum('ijt,jkt,kit->t', S, S, S)

        # Compute Q_W = 1/4 * Ω_i Ω_i
        omega = np.array([
            OMEGA[2, 1, :] - OMEGA[1, 2, :],  # ω_x
            OMEGA[0, 2, :] - OMEGA[2, 0, :],  # ω_y
            OMEGA[1, 0, :] - OMEGA[0, 1, :]   # ω_z
        ])  # Shape: (3, time_int)

        Q_W_part[node_idx, :] = 0.25 * np.sum(omega**2, axis=0)

    return Q_S_part, R_S_part, Q_W_part, block_num

def compute_SQW_vectorized(images_part, block_num):
    """
    Compute the normalized invariants for the rate-of-strain (S) and rate-of-rotation (Ω) tensors.
    
    The procedure is as follows:
    
      1. Compute the fluctuating component of A (velocity gradient tensor):
             A_fluc = A - <A>
         where <A> is the temporal mean for each node.
    
      2. For variance computations, use the fluctuating component:
             S_fluc    = 0.5 * (A_fluc + A_fluc^T)
             Omega_fluc = 0.5 * (A_fluc - A_fluc^T)
         Compute the variances as the time-average of the squared Frobenius norm:
             var_S     = time-average of (S_fluc_ij * S_fluc_ij)
             var_Omega = time-average of (Omega_fluc_ij * Omega_fluc_ij)
    
      3. For computing the invariants, use the full velocity gradient tensor (images_part):
             S_full    = 0.5 * (A + A^T)
             Omega_full = 0.5 * (A - A^T)
         Then compute:
             Qs = -0.5 * (S_full_ij * S_full_ij)
             Rs = -1/3 * (S_full_ij * S_full_jk * S_full_ki)
             Qw =  0.5 * (Omega_full_ij * Omega_full_ij)
    
      4. Normalize the invariants using the variance computed from the fluctuating fields:
             Qs_norm = Qs / (var_S)
             Rs_norm = Rs / (var_S)^(3/2)
             Qw_norm = Qw / var_Omega
    
    Args:
        images_part: np.ndarray of shape (3,3,node_count,time_int), the velocity gradient tensor.
        block_num: Identifier for the data block.
        
    Returns:
        Qs_norm: Normalized second-order invariant for S (shape (node_count, time_int))
        Rs_norm: Normalized third-order invariant for S (shape (node_count, time_int))
        Qw_norm: Normalized second-order invariant for Ω (shape (node_count, time_int))
        block_num: The provided block number (unchanged)
    """
    # Get dimensions
    _, _, node_count, time_int = images_part.shape
    
    # 1. Compute temporal mean and fluctuating component A_fluc
    mean_A = np.mean(images_part, axis=3)              # shape: (3,3,node_count)
    A_fluc = images_part - mean_A[..., None]           # shape: (3,3,node_count,time_int)
    
    # 2. Compute S and Ω for variance (using the fluctuating component)
    S_fluc = 0.5 * (A_fluc + A_fluc.swapaxes(0, 1))     # shape: (3,3,node_count,time_int)
    Omega_fluc = 0.5 * (A_fluc - A_fluc.swapaxes(0, 1))   # shape: (3,3,node_count,time_int)
    
    # Compute variances as time-averaged squared Frobenius norm
    S_fluc_sq = np.sum(S_fluc**2, axis=(0, 1))          # shape: (node_count, time_int)
    var_S = np.mean(S_fluc_sq, axis=1)                  # shape: (node_count,)
    Omega_fluc_sq = np.sum(Omega_fluc**2, axis=(0, 1))    # shape: (node_count, time_int)
    var_Omega = np.mean(Omega_fluc_sq, axis=1)           # shape: (node_count,)
    A_fluc_sq = np.sum(A_fluc**2, axis=(0, 1))          # shape: (node_count, time_int)
    var_A = np.mean(A_fluc_sq, axis=1)                  # shape: (node_count,)
    
    # 3. Compute full S and Ω for invariants (using the full tensor, not the fluctuation)
    S_full = 0.5 * (images_part + images_part.swapaxes(0, 1))    # shape: (3,3,node_count,time_int)
    Omega_full = 0.5 * (images_part - images_part.swapaxes(0, 1))  # shape: (3,3,node_count,time_int)
    
    # Compute invariants from the full tensors:
    # Qs = -0.5 * (S_full_ij * S_full_ij)
    S_full_sq = np.sum(S_full**2, axis=(0, 1))          # shape: (node_count, time_int)
    Qs = -0.5 * S_full_sq                               # shape: (node_count, time_int)
    strain_rate_mean = np.mean(np.sqrt(2 * np.sum(S_full**2, axis=(0, 1))),axis=1)
    # Rs = -1/3 * (S_full_ij * S_full_jk * S_full_ki)
    Rs = -1.0/3.0 * np.einsum('ijnt,jknt,kint->nt', S_full, S_full, S_full)  # shape: (node_count, time_int)
    
    # Qw = 0.5 * (Omega_full_ij * Omega_full_ij)
    Omega_full_sq = np.sum(Omega_full**2, axis=(0, 1))  # shape: (node_count, time_int)
    Qw = 0.5 * Omega_full_sq                            # shape: (node_count, time_int)
    
    # 4. Normalize the invariants using the variance from the fluctuating components
    #Qs_norm = Qs/(strain_rate_mean[:,None])             # normalized with  var_S
    #Qs_norm = Qs / (var_S[:, None])             #   ormalized with  var_S
    #Rs_norm = Rs/(strain_rate_mean[:,None]**3/2)             # normalized with  var_S
    #Rs_norm = Rs / (var_S[:, None]**3/2)         # normalized with var_S
    #Qw_norm = Qw / (strain_rate_mean[:,None])     # normalized with var_Omega
    #Qw_norm = Qw / var_Omega[:, None]     # normalized with var_Omega

    if np.mod(block_num+1, 100) == 0 or block_num == 0:
        print(f'    Processing iteration block number {block_num+1}')
    return Qs, Rs, Qw, var_A, var_S, var_Omega, strain_rate_mean, block_num

def compute_PQR(images_part, block_num):
    """Main worker function to compute the velocity gradient invariants at each node position
    the PQR are invariants of the velocity gradient tensor A_ij = du_i/dx_j
    the invariants are normalized by the temporal variance of the velocity gradient tensor over the time interval
    Args:
        images_part: The velocity gradient tensor at each node position of the partitioned data with size (3, 3, node_count, time_int)
        block_num: The block number of the partitioned data

    Returns:
        P_hat_part: The normalized velocity gradient invariant P_hat at each node position
        Q_hat_part: The normalized velocity gradient invariant Q_hat at each node position
        R_hat_part: The normalized velocity gradient invariant R_hat at each node position
        vort_part: The vorticity magnitude at each node position
        vort_x_part: The x-component of the vorticity vector at each node position
        strain_rate_part: The strain rate magnitude at each node position
        block_num: The block number of the partitioned data
    """
    node_count, time_int = images_part.shape[2], images_part.shape[3]
    # The velocity gradient invariants
    P_hat_part, Q_hat_part, R_hat_part = [np.zeros((node_count, time_int)) for _ in range(3)]
    # The velocity, vortcity, strain rate, and pressure hessian vectors
    vort_part, vort_x_part, strain_rate_part, pressure_hessian_part= [np.zeros((node_count, time_int)) for _ in range(4)]
    if np.mod(block_num+1, 100) == 0 or block_num == 0:
        print(f'    Processing iteration block number {block_num+1}')
    # Compute the velocity gradient invariants at each node
    for node_idx in range(node_count):
        vel_grad = images_part[:, :, node_idx, :]
        mean_A = np.mean(vel_grad, axis=2)
        var_A_time = np.zeros([3, 3, time_int])
        for i in range(time_int):
            A = vel_grad[:, :, i]
            var_A_time[:, :, i] = (A - mean_A) @ np.transpose(A - mean_A)
        # The fluctuating velocity gradient tensor

        # the temporal variance of the velocity gradient tensor for normalization
        var_A = np.mean(var_A_time[0, 0, :] + var_A_time[1, 1, :] + var_A_time[2, 2, :])
        # Initialize the invariants at each time step
        P_hat, Q_hat, R_hat = [np.zeros(time_int) for _ in range(3)]
        vorticity_mag, vort_x_mag ,strain_rate_mag, pressure_hessian_mag = [np.zeros(time_int) for _ in range(4)]
        for i in range(time_int):
            # Compute the velocity gradient invariants
            A = vel_grad[:, :, i]           # The nominal velocity gradient tensor
            A_fluc = A - mean_A
            P_hat[i] = -np.trace(A)
            Q_hat[i] = 0.5 * (np.trace(A)**2 - np.trace(A @ A)) / var_A
            R_hat[i] = -np.linalg.det(A) / var_A**1.5
            # Compute vorticity magnitude
            omega = np.array([
                A[2, 1] - A[1, 2],  # omega_x
                A[0, 2] - A[2, 0],  # omega_y
                A[1, 0] - A[0, 1]   # omega_z
            ])
            vorticity_mag[i] = np.linalg.norm(omega)
            vort_x_mag[i] = omega[0]
            # Compute strain rate magnitude
            S = 0.5 * (A + A.T)         # The strain rate tensor S_ij
            strain_rate_mag[i] = np.sqrt(2 * np.sum(S * S))
            # Compute the pressure hessian
            enstrophy = 0.5 * ((A_fluc[2, 1] - A_fluc[1, 2])**2 + (A_fluc[0, 2] - A_fluc[2, 0])**2 + (A_fluc[1, 0] - A_fluc[0, 1])**2)
            strain = A_fluc[0,0]**2 + A_fluc[1,1]**2 + A_fluc[2,2]**2 + 0.5 * ((A_fluc[0,1] + A_fluc[1,0])**2 + (A_fluc[0,2] + A_fluc[2,0])**2 + (A_fluc[1,2] + A_fluc[2,1])**2)
            pressure_hessian_mag[i] = enstrophy - strain
        # The normalized velocity gradient invariants
        P_hat_part[node_idx, :] = P_hat 
        Q_hat_part[node_idx, :] = Q_hat
        R_hat_part[node_idx, :] = R_hat
        # Store vorticity and strain rate results
        vort_part[node_idx, :] = vorticity_mag
        vort_x_part[node_idx, :] = vort_x_mag
        strain_rate_part[node_idx, :] = strain_rate_mag
        pressure_hessian_part[node_idx, :] = pressure_hessian_mag
    return P_hat_part, Q_hat_part, R_hat_part, vort_part, vort_x_part, strain_rate_part, pressure_hessian_part, block_num


def extract_gradient(arr, cut, reload:bool=False, output:str=None, time:int=None):
    if os.path.exists(output+'/velocity_gradient_tensor_' + cut + '.h5') and not reload:
        print('---->VGT already extracted.')
        print(f'Reloading extracrted velocity gradient tensor from {output}/velocity_gradient_tensor_{cut}.h5')
        with h5py.File(output+'/velocity_gradient_tensor_' + cut + '.h5', 'r') as f:
            velocity = f['velocity'][:]
            velocity_gradient = f['velocity_gradient'][:]
        if time is not None:
            velocity_gradient = velocity_gradient[:, :, :, :time]
            velocity = [v[:, :time] for v in velocity]
            print(f'---->Extracted velocity gradient tensor from h5 files with {time} time steps and {velocity_gradient.shape[2]} nodes.')
    else:
        print('---->Extracting velocity gradient tensor from h5 files.\n')
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

def save_output(node,node_indices,time,results,arr,output,velocity,cut):
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
    print('\n---->Saving results...')
    print('Saving mean PQR')
    # Saving the output mean PQR for visualization
    filename = os.path.join(output,'Velocity_Invarients_' + cut + '_Mean')
    r = Reader('hdf_antares')
    r['filename'] = arr[0]
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
    print('Mean PQR saved to {0:s}.'.format(filename))
    
    print('Saving Full PQR')
    # Saving the full VGT result as h5 file
    output_file = os.path.join(output, 'Velocity_Invarients_' + cut + '.h5')
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
    print('Full PQR saved to {0:s}.'.format(output_file))
    print('\n---->File saving complete.')


def save_output_strain(node,node_indices,time,results,arr,output,velocity,cut):
    # Initialize arrays for final results
    Qs,Rs,Qw = [np.zeros([node, time],dtype='float32') for _ in range(3)]
    var_A, var_S, var_Omega, strain_rate_mean = [np.zeros([node],dtype='float32') for _ in range(4)]
    # Reassemble the global result array
    for Q_S_part, R_S_part, Q_W_part, var_A_part, var_S_part, var_Omega_part, strain_rate_mean_part,block_num in results:
        indices = node_indices[block_num]
        Qs[indices, :],Rs[indices, :], Qw[indices, :] = Q_S_part, R_S_part, Q_W_part
        var_A[indices], var_S[indices], var_Omega[indices], strain_rate_mean[indices] = var_A_part, var_S_part, var_Omega_part, strain_rate_mean_part
    del results         # Free up memory
    print('\n---->Saving results...')
    print('Saving mean Qs, Rs, Qw')
    # Saving the output mean PQR for visualization
    filename = os.path.join(output,'Velocity_Invarients_Rotation_Strain_' + cut + '_Mean')
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
    w = Writer('hdf_antares')  # This is another format (still hdf5)
    w['base'] = b[:, :, ['x', 'y', 'z', 'Qs', 'Rs', 'Qw', 'Variance Qs', 'Variance Rs', 'Variance Qw','Variance A','Variance S','Variance Omega','Mean strain_rate']]
    w['filename'] = filename
    w.dump()
    print('Mean Qs, Rs, Qw saved to {0:s}.'.format(filename))
    
    print('Saving Full Qs, Rs, Qw')
    # Saving the full VGT result as h5 file
    output_file = os.path.join(output, 'Velocity_Invarients_' + cut + '.h5')
    with h5py.File(output_file, 'a') as f:
        f.create_dataset('Qs', data=Qs, dtype='float32')
        f.create_dataset('Rs', data=Rs, dtype='float32')
        f.create_dataset('Qw', data=Qw, dtype='float32')
        f.create_dataset('Variance A', data=var_A, dtype='float32')
        f.create_dataset('Variance S', data=var_S, dtype='float32')
        f.create_dataset('Variance Omega', data=var_Omega, dtype='float32')
        f.create_dataset('Mean strain_rate', data=strain_rate_mean, dtype='float32')
    print('Full Qs, Rs, Qw appended to {0:s}.'.format(output_file))
    print('\n---->File saving complete.')

@timer
def main():
    text = 'Performing velocity gradient invariants computation.'
    print(f'\n{text:.^120}\n')  
    global velocity_gradient
    cut = argv[1]
    reload = False
    file_main = os.getcwd()
    file_dir = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface/Cut_' + cut + '_VGT/*.h5'
    file_dir = os.path.join(file_main, file_dir)
    arr = sorted(glob.glob(file_dir))
    
    # Create an output directory if it does not exist
    output = 'Velocity_Invarients_'+cut
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    
    # Extract the velocity gradient tensor from the cutplane data
    velocity_gradient_np, velocity = extract_gradient(arr, cut, reload, output, time=None)
    
    # Partition the data into blocks for parallel computation
    node, time = velocity_gradient_np.shape[2], velocity_gradient_np.shape[3]
    num_processes = cpu_count()-20  # Limit to number of processes
    nb_tasks = 1000  # Number of tasks for splitting data
    node_indices = np.array_split(np.arange(node), nb_tasks)
    images_parts = [(velocity_gradient_np[:, :, indices, :], block_num) for block_num, indices in enumerate(node_indices)]
    print('\n---->Partitioning data into {0:d} blocks.'.format(len(node_indices)))
    print(f'Number of available parallel compute processes: {num_processes}')
    print(f'Number of nodes: {node}')
    print(f'Number of time steps: {time}')
    print('\n----> Perofrming Parallel VGT computations...')
    del velocity_gradient_np  # Free up memory
    
    # Parallel computation of the velocity gradient invariants
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_PQR_vectorized, images_parts)
    #images_parts  # Free up memory
    print('\n---->Velocity gradient invariants computation complete.')
    
    # Save the output
    save_output(node, node_indices, time, results, arr, output, velocity, cut)
    del results
    
    print('\n----> Perofrming Parallel Strain and Vorticity Tensor computations...')
    # Parallel computation of the rate-of-strain and rate-of-rotation invariants
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_SQW_vectorized, images_parts)
    del images_parts
    save_output_strain(node, node_indices, time, results, arr, output, velocity, cut)
    print('\n---->Rate-of-strain and rate-of-rotation invariants computation complete.')
    
    text = 'Velocity gradient invariants computation complete.'
    print(f'\n{text:.^120}\n')  

if __name__ == '__main__':
    main()
