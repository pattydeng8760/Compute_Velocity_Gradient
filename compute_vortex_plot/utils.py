import os
import sys
import time
import logging
import numpy as np
from functools import wraps
from scipy.signal import welch
from scipy.signal.windows import hann
import matplotlib.pyplot as plt

def print(*args, **kwargs):
    """Custom print function that also logs to file."""
    # If logging is configured, use logging (which handles both file and console)
    if logging.getLogger().hasHandlers():
        message = ' '.join(str(arg) for arg in args)
        logging.info(message)
    else:
        # If no logging configured, use built-in print
        import builtins
        builtins.print(*args, **kwargs)

def init_logging_from_cut(cut, data_type='LES'):
    """Initialize logging and redirect stdout to a log file."""
    log_filename = f"log_vortex_plot_{cut}_{data_type}.txt"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    sys.stdout = open(log_filename, "w", buffering=1)
    # # Set up logging configuration
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(message)s',
    #     handlers=[
    #         logging.FileHandler(log_filename),
    #         logging.StreamHandler(sys.stdout)
    #     ]
    # )
    # print(f"Logging initialized. Output will be saved to: {log_filename}")

def timer(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"The total compute time is: {int(time.time() - start)} s")
        return result
    return inner

def _next_greater_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())

def _welch_psd(x, dt, nchunk:int=1):
    x = np.asarray(x).ravel()
    fs = 1.0 / dt
    lensg = len(x)
    nperseg = int(lensg / nchunk)
    nfft = _next_greater_power_of_2(nperseg)   
    f, Pxx = welch(x, fs=fs, window='hamming', nperseg=nperseg, nfft=nfft, scaling='density')
    return f, Pxx


def _setup_plot_params():
    """Setup matplotlib parameters for consistent plot styling."""
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    LARGE_SIZE = 22
    
    plt.rcParams.update({
        'font.size': MEDIUM_SIZE,
        'axes.titlesize': MEDIUM_SIZE,
        'axes.labelsize': MEDIUM_SIZE,
        'xtick.labelsize': MEDIUM_SIZE,
        'ytick.labelsize': MEDIUM_SIZE,
        'legend.fontsize': SMALL_SIZE,
        'figure.titlesize': LARGE_SIZE,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
    })