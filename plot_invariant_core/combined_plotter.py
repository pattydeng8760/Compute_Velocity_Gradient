import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from .data_loader import extract_qr_data
from .utils import setup_rcparams, print, get_vortex_locations

def create_combined_qr_plots(data_file, output_dir, chord=0.305):
    """Create combined QR plots for all vortex types across locations."""
    setup_rcparams()
    
    # Define locations and vortex types
    locations = get_vortex_locations()
    vortex_types = ['PV', 'SV', 'TV', 'SS_shear', 'PS_shear']
    num_features = None  # Extract the max number of datapoints without truncation
    
    print(f"    Creating combined QR plots for locations: {locations}")
    print(f"    Processing vortex types: {vortex_types}")
    
    # Process each vortex type
    for vortex_type in vortex_types:
        print(f"    Processing {vortex_type} vortex type...")
        
        try:
            # Extract data for this vortex type
            q, qs, qw, r, rs = extract_qr_data(data_file, locations, vortex_type, num_features)
            
            if len(q) == 0 or len(r) == 0:
                print(f"        No data found for {vortex_type}, skipping...")
                continue
            
            # Generate QR plots
            plot_qr_along_vortex(q, r, bins=100, vortex_group=vortex_type, output_dir=output_dir)
            plot_qs_rs_along_vortex(qs, rs, bins=100, vortex_group=vortex_type, output_dir=output_dir)
            plot_qs_qw_along_vortex(qs, qw, bins=100, vortex_group=vortex_type, output_dir=output_dir)
            
            print(f"        Generated QR, Qs-Rs, and Qs-Qw plots for {vortex_type}")
            
        except Exception as e:
            print(f"        Error processing {vortex_type}: {e}")
            continue

def plot_qr_along_vortex(q, r, bins=100, set_lim=None, vortex_group='PV', output_dir='.'):
    """Plot Q-R relationship along vortex locations."""
    if len(q) == 0 or len(r) == 0:
        print(f"        No data to plot for {vortex_group} QR")
        return
    
    # Create figure and axes
    fig, axs = plt.subplots(1, int(r.shape[0]), figsize=(int(r.shape[0]*2), 3), 
                           sharey=True, gridspec_kw={'wspace': 0})
    
    # Ensure axs is iterable
    if r.shape[0] == 1:
        axs = [axs]
    
    # Loop through each position and plot the 2D histogram
    for i in range(r.shape[0]):
        Rhat = r[i, :]
        Qhat = q[i, :]
        
        # Compute the 2D histogram
        h, edgesR, edgesQ = np.histogram2d(Rhat, Qhat, bins=bins)
        
        # Normalize the histogram
        totalPoints = Rhat.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Center the bin edges
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)
        
        # Set axis limits
        if set_lim is not None:
            axs[i].set_xlim(set_lim[0], set_lim[1])
            axs[i].set_ylim(set_lim[2], set_lim[3])
        else:
            axs[i].set_xlim(-0.25, 0.25)
            axs[i].set_ylim(-1, 1)
        
        # Normalize and apply Gaussian filter
        pdf_norm = pdf / np.max(pdf)
        pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
        
        # Delta=0 curve
        x_zero_pdf = np.linspace(-2, 2, 1000)
        Q_zero_pdf = -3 * ((x_zero_pdf / 2) ** 2) ** (1 / 3)
        
        # Plot the contour
        color_map = plt.cm.hot.reversed()
        c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                           cmap=color_map, extend='both')
        axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                      colors='blue', linestyles='dashed', linewidths=1)
        
        # Add reference lines
        axs[i].axvline(0, color='black', linestyle='--', linewidth=2)
        axs[i].plot(x_zero_pdf, Q_zero_pdf, '--k', linewidth=2)
        
        # Set labels
        if i < r.shape[0] - 1:
            axs[i].set_xticklabels([])
        axs[i].set_xlabel('$\hat{R}$')
    
    axs[0].set_ylabel('$\hat{Q}$')
    fig.tight_layout()
    
    # Save plot
    figname = f'{output_dir}/B_10AOA_{vortex_group}_Cores_horizontal'
    plt.savefig(figname + '.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
    plt.savefig(figname + '.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_qs_rs_along_vortex(qs, rs, bins=100, set_lim=None, vortex_group='PV', output_dir='.'):
    """Plot Qs-Rs relationship along vortex locations."""
    if len(qs) == 0 or len(rs) == 0:
        print(f"        No data to plot for {vortex_group} QsRs")
        return
    
    # Create figure and axes
    fig, axs = plt.subplots(1, int(rs.shape[0]), figsize=(int(rs.shape[0]*2), 3), 
                           sharey=True, gridspec_kw={'wspace': 0})
    
    # Ensure axs is iterable
    if rs.shape[0] == 1:
        axs = [axs]
    
    a_values = [-0.5, 0, 1/2, 1/3, 1]
    
    # Loop through each position and plot the 2D histogram
    for i in range(rs.shape[0]):
        Rhat = rs[i, :]
        Qhat = qs[i, :]
        
        # Compute the 2D histogram
        h, edgesR, edgesQ = np.histogram2d(Rhat, Qhat, bins=bins)
        
        # Normalize the histogram
        totalPoints = Rhat.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Center the bin edges
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)
        
        # Set axis limits
        if set_lim is not None:
            axs[i].set_xlim(set_lim[0], set_lim[1])
            axs[i].set_ylim(set_lim[2], set_lim[3])
        else:
            axs[i].set_xlim(-0.5, 0.5)
            axs[i].set_ylim(-1.5, 0)
        
        # Normalize and apply Gaussian filter
        pdf_norm = pdf / np.max(pdf)
        pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
        
        # Plot the contour
        color_map = plt.cm.hot.reversed()
        c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                           cmap=color_map, extend='both')
        axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                      colors='blue', linestyles='dashed', linewidths=1)
        
        # Add theoretical lines
        Q_plus = np.linspace(0.001, 1.5, 1000)
        for a in a_values:
            RS_line = - (Q_plus**(3/2)) * a * (1+a) / ((1+a+a**2)**(3/2))
            axs[i].plot(-RS_line, -Q_plus, '--', linewidth=2)
        
        # Set labels
        if i < rs.shape[0] - 1:
            axs[i].set_xticklabels([])
        axs[i].set_xlabel('$\hat{R}_s$')
    
    axs[0].set_ylabel('$\hat{Q}_s$')
    fig.tight_layout()
    
    # Save plot
    figname = f'{output_dir}/B_10AOA_{vortex_group}_Cores_Qs_Rs_horizontal'
    plt.savefig(figname + '.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(figname + '.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_qs_qw_along_vortex(qs, qw, bins=100, set_lim=None, vortex_group='PV', output_dir='.'):
    """Plot Qs-Qw relationship along vortex locations."""
    if len(qs) == 0 or len(qw) == 0:
        print(f"        No data to plot for {vortex_group} QsQw")
        return
    
    # Create figure and axes
    fig, axs = plt.subplots(1, int(qs.shape[0]), figsize=(int(qs.shape[0]*2), 3), 
                           sharey=True, gridspec_kw={'wspace': 0})
    
    # Ensure axs is iterable
    if qs.shape[0] == 1:
        axs = [axs]
    
    # Loop through each position and plot the 2D histogram
    for i in range(qs.shape[0]):
        Rhat = qw[i, :]
        Qhat = -qs[i, :]
        
        # Define the desired ranges
        Rmin, Rmax = 0, np.max(Rhat)
        Qmin, Qmax = np.min(Qhat), np.max(Qhat)
        
        # Compute the 2D histogram
        h, edgesR, edgesQ = np.histogram2d(Rhat, Qhat, bins=bins, range=[[Rmin, Rmax], [Qmin, Qmax]])
        
        # Normalize the histogram
        totalPoints = Rhat.size
        binArea = np.diff(edgesR[:2])[0] * np.diff(edgesQ[:2])[0]
        pdf = h / totalPoints / binArea
        
        # Center the bin edges
        edgesR_centered = edgesR[:-1] + np.diff(edgesR) / 2
        edgesQ_centered = edgesQ[:-1] + np.diff(edgesQ) / 2
        x_pdf, y_pdf = np.meshgrid(edgesR_centered, edgesQ_centered)
        
        # Set axis limits
        if set_lim is not None:
            axs[i].set_xlim(set_lim[0], set_lim[1])
            axs[i].set_ylim(set_lim[2], set_lim[3])
        else:
            axs[i].set_xlim(0, 1.5)
            axs[i].set_ylim(0, 1.5)
        
        # Normalize and apply Gaussian filter
        pdf_norm = pdf / np.max(pdf)
        pdf_norm = gaussian_filter(pdf_norm, sigma=[2.5, 2.5])
        
        # Plot the contour
        color_map = plt.cm.hot.reversed()
        c = axs[i].contourf(x_pdf, y_pdf, pdf_norm.T, levels=np.linspace(0, 0.3, 64), 
                           cmap=color_map, extend='both')
        axs[i].contour(x_pdf, y_pdf, pdf_norm.T, levels=[0.025, 0.08, 0.13, 0.2], 
                      colors='blue', linestyles='dashed', linewidths=1)
        
        # Add reference line
        x_zero_pdf = np.linspace(x_pdf[0, 0], x_pdf[0, -1], 1000)
        Q_zero_pdf = x_zero_pdf
        axs[i].plot(x_zero_pdf, Q_zero_pdf, '--k', linewidth=2, zorder=10)
        
        # Set labels
        if i < qs.shape[0] - 1:
            axs[i].set_xticklabels([])
        axs[i].set_xlabel('$\hat{Q}_w$')
    
    axs[0].set_ylabel('$-\hat{Q}_s$')
    fig.tight_layout()
    
    # Save plot
    figname = f'{output_dir}/B_10AOA_{vortex_group}_Cores_Qw_Qs_horizontal'
    plt.savefig(figname + '.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(figname + '.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
    plt.close(fig)