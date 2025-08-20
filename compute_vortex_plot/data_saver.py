import h5py
from .combine_qr_plot import get_data_file_path
def save_extracted_data(location, P_hat, Q_hat, R_hat, Qs_hat, Qw_hat, Rs_hat, Vortex_Type, data_type='LES', velocity=30, angle_of_attack=10, limited_gradient=False):
    """
    Save extracted velocity invariant data to HDF5 file.
    
    Parameters:
    -----------
    location : str
        Location identifier
    P_hat : array
        P invariant data
    Q_hat : array
        Q invariant data
    R_hat : array
        R invariant data
    Qs_hat : array
        Qs invariant data
    Qw_hat : array
        Qw invariant data
    Rs_hat : array
        Rs invariant data
    Vortex_Type : str
        Vortex type identifier
    data_type : str
        Data type ('LES' or 'PIV')
    """

    filename = get_data_file_path(data_type, velocity=velocity, angle_of_attack=angle_of_attack, limited_gradient=limited_gradient, check_exists=False)
    
    with h5py.File(filename, 'a') as f:
        # Ensure the location group exists
        if location not in f:
            f.create_group(location)
        
        # Create the Vortex_Type subgroup under the location group if it does not already exist
        if Vortex_Type not in f[location]:
            f[location].create_group(Vortex_Type)
        
        # Save each dataset under the Vortex_Type subgroup
        for name, dat in zip(['P', 'Q', 'R', 'Qs', 'Qw', 'Rs'],
                             [P_hat, Q_hat, R_hat, Qs_hat, Qw_hat, Rs_hat]):
            # If the dataset already exists, remove it
            if name in f[location][Vortex_Type]:
                del f[location][Vortex_Type][name]
            f[location][Vortex_Type].create_dataset(name, data=dat)