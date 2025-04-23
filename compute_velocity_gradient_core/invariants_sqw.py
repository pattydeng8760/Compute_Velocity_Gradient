import numpy as np
from .utils import print

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


# THE following function is depricated and not used in the code, replaced by the vectorized version above
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