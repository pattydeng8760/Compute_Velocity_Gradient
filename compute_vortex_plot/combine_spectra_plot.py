import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from .utils import print, _next_greater_power_of_2, _welch_psd

def extract_slices_with_dt(
    data_file, locations,vortices, variables=('u','v','w','pressure','vort_x'), num_features=825):
    out = {loc: {vx: {var: {'row': None, 'dt': None, 'present': False} for var in variables} for vx in vortices}
           for loc in locations}
    with h5py.File(data_file, 'r') as f:
        for loc in locations:
            has_loc = loc in f
            for vx in vortices:
                has_group = has_loc and (vx in f[loc])
                subgroup = f[loc][vx] if has_group else None
                dt_attr = None
                if has_group:
                    dt_attr = subgroup.attrs.get('dt', 1.0)
                    try:
                        dt_attr = float(dt_attr)
                    except Exception:
                        dt_attr = 1.0
                for var in variables:
                    if has_group and (var in subgroup):
                        row = subgroup[var][0]
                        row = row[:num_features] if row.shape[0] >= num_features else np.pad(row, (0, num_features - row.shape[0]))
                        out[loc][vx][var] = {'row': row, 'dt': dt_attr, 'present': True}
                    else:
                        out[loc][vx][var] = {'row': np.zeros(num_features), 'dt': (dt_attr if dt_attr is not None else 1.0), 'present': False}
    return out

def _make_touching_row(nloc):
    fig, axes = plt.subplots(1, nloc, sharey=True, figsize=(4*nloc, 3.2))
    if nloc == 1:
        axes = np.array([axes])
    fig.set_constrained_layout(False)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.12, wspace=0.0, hspace=0.0)
    return fig, axes

# ---------------- Function to add -5/3 slope line ----------------
def _add_ref_slope(ax, f1=1e3, f2=1e4, y_mean=None):
    """
    Add a -5/3 reference slope between f1 and f2.
    The line starts one order of magnitude below y_mean at f1,
    and the label is drawn directly on the figure.
    """
    x_ref = np.array([f1, f2])
    
    # Pick anchor level
    if y_mean is None:
        ymin, ymax = ax.get_ylim()
        if ymin <= 0:
            ymin = np.nextafter(0, 1)
        y_mean = 10**((np.log10(ymin) + np.log10(ymax)) / 2)

    # anchor 1 order of magnitude lower
    y1 = y_mean / 10.0
    y2 = y1 * (f2 / f1)**(-5/3)

    y_ref = np.array([y1, y2])
    ax.plot(x_ref, y_ref, 'k--', lw=1)

    # Add slope text near the middle of the line
    x_mid = np.sqrt(f1 * f2)                   # geometric mean for midpoint in log space
    y_mid = y1 * (x_mid / f1)**(-5/3)          # corresponding y value

    # Add text below the line (scale factor < 1 moves it below)
    ax.text(x_mid, y_mid*0.7, r"$f^{-5/3}$", 
            ha='center', va='top', rotation=0)


def plot_combine_spectra(data_file, locations, vortices=('PV','SV','TV'), variables=('u','v','w','pressure','vort_x'),
    num_features=825, nchunk=1, output_dir=None, plot_ref_slope=True, AOA:int=10, U_inf:int=30):
    # --- mapping for location labels ---
    LOC_LABELS = {
        "030_TE": "$x/c$=0.30",
        "085_TE": "$x/c$=0.85",
        "PIV1":   "$x/c$=0.58",
        "PIV2":   "$x/c$=0.78",
        "095_TE": "$x/c$=0.95",
        "PIV3":   "$x/c$=1.05",
    }
    YLABELS = {
        "u":      r"$\Phi_{uu}$ [m$^2$s$^{-2}$/Hz]",
        "v":      r"$\Phi_{vv}$ [m$^2$s$^{-2}$/Hz]",
        "w":      r"$\Phi_{ww}$ [m$^2$s$^{-2}$/Hz]",
        "pressure": r"$\Phi_{pp}$ [dB ref. $20 \mu$Pa]",
        "vort_x": r"$\Phi_{\omega_x\omega_x}$ [s$^{-2}$/Hz]",
        "TKE":    r"$E_k$ [m$^2$s$^{-2}$/Hz]",
    }
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    data = extract_slices_with_dt(
        data_file, list(locations), list(vortices), list(variables), num_features=num_features
    )

    # ---- Per-variable figures (legend only on last subplot) ----
    for var in variables:
        fig, axes = _make_touching_row(len(locations))
        for i, loc in enumerate(locations):
            ax = axes[i]
            # plot all vortices (available) for this location
            for vx in vortices:
                entry = data[loc][vx][var]
                if not entry['present']:
                    continue
                f, Pxx = _welch_psd(entry['row'], dt=entry['dt'], nchunk=nchunk)
                if var == "pressure":
                    ref = 2e-5
                    Pxx_db = 10 * np.log10(Pxx / (ref**2) + 1e-30)
                    ax.plot(f, Pxx_db, label=vx)
                else:
                    ax.plot(f, Pxx, label=vx)

            # add slope reference (only for log-log plots)
            if plot_ref_slope and var in ("u","v","w","vort_x"):
                # compute geometric mean of PSD values for anchoring
                all_vals = []
                for vx in vortices:
                    entry = data[loc][vx][var]
                    if entry['present']:
                        f, Pxx = _welch_psd(entry['row'], dt=entry['dt'], nchunk=nchunk)
                        all_vals.append(Pxx)
                if all_vals:
                    mean_val = np.mean(np.concatenate(all_vals))
                    _add_ref_slope(ax, y_mean=mean_val)

            # lower-left location label text (no title)
            loc_text = LOC_LABELS.get(loc, loc)
            ax.text(0.95, 0.95, loc_text, transform=ax.transAxes,
                    ha='right', va='top')

            ax.grid(True, which="both", linestyle=":")
            # scales
            if var in ("u","v","w","vort_x"):
                ax.set_xscale("log"); ax.set_yscale("log")
            elif var == "pressure":
                ax.set_xscale("log"); ax.set_yscale("linear")
            # labels
            ax.set_xlabel("Frequency [Hz]")
            if i == 0:
                ax.set_ylabel(YLABELS[var])

        # Legend only in the last subplot of this figure
        last_ax = axes[-1]
        handles, labels = last_ax.get_legend_handles_labels()
        if handles:
            last_ax.legend(handles, labels, loc="lower left", frameon=False)

        fig.suptitle(f"{var} — Spectra ", y=0.99)
        filename = f"B_{AOA}AOA_U{U_inf}_spectra_{var}"
        fig.savefig(os.path.join(output_dir, filename + ".png"), dpi=200)
        plt.close(fig)
    # ---- TKE figure ----
    fig, axes = _make_touching_row(len(locations))
    for i, loc in enumerate(locations):
        ax = axes[i]
        # compute TKE PSD per vortex
        for vx in vortices:
            e_u, e_v, e_w = data[loc][vx]['u'], data[loc][vx]['v'], data[loc][vx]['w']
            if not (e_u['present'] or e_v['present'] or e_w['present']):
                continue
            parts, f_ref = [], None
            for comp in (e_u, e_v, e_w):
                if comp['present']:
                    f, P = _welch_psd(comp['row'], dt=comp['dt'], nchunk=nchunk)
                    if f_ref is None:
                        f_ref, parts = f, [P]
                    elif not np.array_equal(f, f_ref):
                        P = np.interp(f_ref, f, P, left=0.0, right=0.0)
                        parts.append(P)
                    else:
                        parts.append(P)
            if f_ref is None:
                continue
            P_tke = 0.5 * np.sum(parts, axis=0)
            ax.plot(f_ref, P_tke, label=vx)

        # lower-left location label text (no title)
        loc_text = LOC_LABELS.get(loc, loc)
        ax.text(0.95, 0.95, loc_text, transform=ax.transAxes,
                ha='right', va='top')

        ax.grid(True, which="both", linestyle=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Frequency [Hz]")
        if i == 0:
            ax.set_ylabel("$E_k$ [m$^2$s$^{-2}$/Hz]")
        if plot_ref_slope:
            mean_val = np.mean(P_tke) if P_tke is not None else None
            _add_ref_slope(ax, y_mean=mean_val)

    # Legend only in the last TKE subplot
    last_ax = axes[-1]
    handles, labels = last_ax.get_legend_handles_labels()
    if handles:
        last_ax.legend(handles, labels, loc="lower left", frameon=False)

    fig.suptitle("TKE — Spectra", y=0.99)
    filename = f"B_{AOA}AOA_U{U_inf}_spectra_TKE"
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename + ".png"), dpi=600)
    plt.close(fig)