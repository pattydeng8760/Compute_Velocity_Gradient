import numpy as np
from antares import *

"""
Compute the Turbulent Kinetic Energy (TKE) balance from the velocity gradient tensor.
This module computes the TKE balance using the velocity gradient tensor components.
"""

def calculate_tke(velocity):
    """
    Calculate turbulent kinetic energy defined as: tke = 0.5 * (u'² + v'² + w'²)
    
    Args:
        velocity (list): List of 3 velocity components [u, v, w] with shape (nodes, time)
    
    Returns:
        np.array: TKE with shape (nodes, time)
    """
    u, v, w = velocity
    tke_array = np.zeros((u.shape[0], u.shape[1]), dtype='float32')
    # Time-averaged mean velocities
    u_mean = np.mean(u, axis=1, keepdims=True)
    v_mean = np.mean(v, axis=1, keepdims=True)
    w_mean = np.mean(w, axis=1, keepdims=True)
    
    # Calculate fluctuating velocities
    u_prime = u - u_mean
    v_prime = v - v_mean
    w_prime = w - w_mean
    
    # Calculate TKE = 0.5 * (u'² + v'² + w'²)
    tke_array = 0.5 * (u_prime**2 + v_prime**2 + w_prime**2)
    
    return tke_array


def calculate_reynolds_stress(velocity):
    """
    Calculate mean Reynolds stress components from velocity time series.

    Args:
        velocity (list): List of 3 velocity components [u, v, w], each with shape (nodes, time)

    Returns:
        list: Mean Reynolds stress components [R_uu, R_vv, R_ww, R_uv, R_uw, R_vw], each (nodes,)
    """
    u, v, w = velocity

    # Compute time-averaged velocities
    u_mean = np.mean(u, axis=1, keepdims=True)
    v_mean = np.mean(v, axis=1, keepdims=True)
    w_mean = np.mean(w, axis=1, keepdims=True)

    # Compute fluctuations: shape (nodes, time)
    u_prime = u - u_mean
    v_prime = v - v_mean
    w_prime = w - w_mean

    # Compute mean Reynolds stress components (over time)
    R_uu = np.mean(u_prime * u_prime, axis=1)
    R_vv = np.mean(v_prime * v_prime, axis=1)
    R_ww = np.mean(w_prime * w_prime, axis=1)
    R_uv = np.mean(u_prime * v_prime, axis=1)
    R_uw = np.mean(u_prime * w_prime, axis=1)
    R_vw = np.mean(v_prime * w_prime, axis=1)
    reynolds_stress_array = [R_uu, R_vv, R_ww, R_uv, R_uw, R_vw]
    return reynolds_stress_array

def calcualte_tke_convection(velocity, tke_array, data_type:str='PIV', arr:list=None):
    """
    Calculate the TKE convection term from the velocity gradient tensor.
    The convection term is calculated as:
    d(tke)/dt = - u_i * grad_i(tke) = - (mean(u) * grad_u(tke) + mean(v) * grad_v(tke) + mean(w) * grad_w(tke))
    """
    u, v, w = velocity
    #du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz = velocity_gradient
    # Time-averaged mean velocities
    u_mean = np.mean(u, axis=1)
    v_mean = np.mean(v, axis=1)
    w_mean = np.mean(w, axis=1)
    # For PIV coordinates = X: wall normal, Y: spanwise, Z: streamwise
    # For LES coordiantes = X: streamwise, Y: wall normal, Z: spanwsie
    if data_type == 'PIV':
        r = Reader('hdf_antares')
        r['filename'] = arr[0]
        b = r.read() 
        dy = b[0][0].attrs['dx_piv'] 
        dz = b[0][0].attrs['dy_piv']
        dk_dy, dk_dz = np.gradient(tke_array,dy,dz,edge_order=2)
        rho = 1.225  # Assuming constant density of air at sea level
    convection = np.zeros_like(tke_array, dtype='float32')
    # Calculate the convection term for the TKE balance
    convection = - rho*(v_mean * dk_dz + w_mean * dk_dz)
    return convection

def calculate_tke_diffusion(velocity,tke_array, velocity_gradient, data_type:str='PIV', arr:list=None):
    """
    Calculate the TKE diffusion term from the velocity gradient tensor.
    The diffusion term is calculated as:
    div(u' * grad(tke)) 
    """
    u, v, w = velocity
    # Time-averaged mean velocities
    u_mean, v_mean, w_mean = np.mean(u, axis=1), np.mean(v, axis=1), np.mean(w, axis=1)
    u_prime, v_prime, w_prime = u - u_mean, v - v_mean, w - w_mean
    k_prime = tke_array - np.mean(tke_array, axis=1)
    if data_type == 'PIV':
        r = Reader('hdf_antares')
        r['filename'] = arr[0]
        b = r.read() 
        rho = 1.225
        dy = b[0][0].attrs['dx_piv'] 
        dz = b[0][0].attrs['dy_piv']
        tdiff_x, tdiff_y, tdiff_z = np.mean(u_prime+rho*k_prime, axis=1), np.mean(v_prime+rho*k_prime, axis=1) ,np.mean(w_prime+rho*k_prime, axis=1) 
        dk_dy, _ = np.gradient(tdiff_y, dy, dz, edge_order=2)
        _, dk_dz = np.gradient(tdiff_z, dy, dz, edge_order=2)
    diffusion = np.zeros_like(tke_array, dtype='float32')
    # Calculate the diffusion term for the TKE balance
    diffusion = dk_dy + dk_dz
    return diffusion

def calculate_tke_production(velocity, velocity_gradient, reynolds_stress_array):
    """
    Calculate the TKE production term from the Reynolds stress tensor.
    The production term is calculated as:
    P = - u_i * grad_i(R_ij)
    """
    u, v, w = velocity
    # The components of the velocity gradient tensor
    du_dx, du_dy, du_dz = velocity_gradient[0], velocity_gradient[1], velocity_gradient[2]
    dv_dx, dv_dy, dv_dz = velocity_gradient[3], velocity_gradient[4], velocity_gradient[5]
    dw_dx, dw_dy, dw_dz = velocity_gradient[6], velocity_gradient[7], velocity_gradient[8]
    
    # The mean gradient tensor components
    du_dx_mean,du_dy_mean,du_dz_mean = np.mean(du_dx, axis=1), np.mean(du_dy, axis=1), np.mean(du_dz, axis=1)
    dv_dx_mean,dv_dy_mean,dv_dz_mean = np.mean(dv_dx, axis=1), np.mean(dv_dy, axis=1), np.mean(dv_dz, axis=1)
    dw_dx_mean,dw_dy_mean,dw_dz_mean = np.mean(dw_dx, axis=1), np.mean(dw_dy, axis=1), np.mean(dw_dz, axis=1)
    
    # Extract Reynolds stress components
    R_uu, R_vv, R_ww, R_uv, R_uw, R_vw = reynolds_stress_array
    
    # Calculate the production term for the TKE balance
    production, production_normal, production_shear = [np.zeros_like(R_uu, dtype='float32')for _ in range(3)]
    production_normal = R_uu*du_dx_mean + R_vv*dv_dy_mean + R_ww*dw_dz_mean 
    production_shear = R_uv*(du_dy_mean + dv_dx_mean) + R_uw*(du_dz_mean + dw_dx_mean) + R_vw*(dv_dz_mean + dw_dy_mean)
    production = production_normal + production_shear
    
    return production, production_normal, production_shear

def calculate_tke_dissipation(velocity, velocity_gradient):
    """
    Calculate the TKE dissipation term from the velocity gradient tensor.
    The dissipation term is calculated as:
    D = 2 * nu * S_ij * S_ij
    where S_ij = 0.5 * (grad_i(u_j) + grad_j(u_i))
    """
    u, v, w = velocity
    du_dx, du_dy, du_dz = velocity_gradient[0], velocity_gradient[1], velocity_gradient[2]
    dv_dx, dv_dy, dv_dz = velocity_gradient[3], velocity_gradient[4], velocity_gradient[5]
    dw_dx, dw_dy, dw_dz = velocity_gradient[6], velocity_gradient[7], velocity_gradient[8]
    # The strain components of the velocity gradient tensor
    S_xx, S_yy, S_zz = np.mean(du_dx*du_dx, axis=1), np.mean(dv_dy*dv_dy, axis=1), np.mean(dw_dz*dw_dz, axis=1)
    S_xy, S_xz, S_yz = 0.5*np.mean(du_dy + dv_dx, axis=1), 0.5*np.mean(du_dz + dw_dx, axis=1), 0.5*np.mean(dv_dz + dw_dy, axis=1)
    mu = 1.81e-5  # Dynamic viscosity of air at sea level
    # Calculate the dissipation term for the TKE balance
    dissipation = 2 * mu * (S_xx + S_yy + S_zz + (S_xy + S_xz + S_yz))
    dissipation_normal = 2 * mu * (S_xx + S_yy + S_zz)
    dissipation_shear = 2 * mu * (S_xy + S_xz + S_yz)
    return dissipation, dissipation_normal, dissipation_shear


def tke_balance(velocity, velocity_gradient):
    """
    Main function to calcualte the TKE balance from the subfunctions
    
    """