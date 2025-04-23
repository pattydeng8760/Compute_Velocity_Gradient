import numpy as np
from .utils import print

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


# The following function is deprecated and replaced by the vectorized version above.
def compute_PQR_mod(images_part, block_num):
    """Main worker function to compute the velocity gradient invariants at each node position
    the PQR are invariants of the velocity gradient tensor A_ij = du_i/dx_j
    the invariants are normalized by the temporal variance of the velocity gradient tensor over the time interval
    NOTE: THIS FUNCTION IS DEPRECATED AND REPLACED BY THE VECTORIZED VERSION ABOVE
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


# The following function is deprecated and replaced by the vectorized version above.
def compute_PQR(images_part, block_num):
    """Main worker function to compute the velocity gradient invariants at each node position
    the PQR are invariants of the velocity gradient tensor A_ij = du_i/dx_j
    the invariants are normalized by the temporal variance of the velocity gradient tensor over the time interval
    NOTE: THIS FUNCTION IS DEPRECATED AND REPLACED BY THE VECTORIZED VERSION ABOVE
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