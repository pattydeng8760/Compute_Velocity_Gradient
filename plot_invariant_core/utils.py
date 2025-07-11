import os
import sys
import time
import logging
from functools import wraps
import matplotlib.pyplot as plt

def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.2f} seconds to execute.")
        return result
    return wrapper

def print(*args, **kwargs):
    """Custom print function that also logs to file."""
    # Print to stdout
    __builtins__['print'](*args, **kwargs)
    
    # Log to file if logging is configured
    if logging.getLogger().hasHandlers():
        message = ' '.join(str(arg) for arg in args)
        logging.info(message)

def init_logging_from_cut(cut, data_type='LES'):
    """Initialize logging and redirect stdout to a log file."""
    log_filename = f"log_plot_invariants_{cut}_{data_type}.txt"
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print(f"Logging initialized. Output will be saved to: {log_filename}")

def setup_rcparams():
    """Set up matplotlib rcParams for consistent plot appearance."""
    SMALL_SIZE = 12
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

def validate_file_exists(file_path, description="File"):
    """Validate that a file exists and raise an error if it doesn't."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found: {file_path}")
    return True

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_vortex_types():
    """Get the list of vortex types to process."""
    return ['PV', 'SV', 'TV', 'SS_shear', 'PS_shear']

def get_vortex_locations():
    """Get the list of vortex locations to process."""
    return ['030_TE', 'PIV1', 'PIV2', '085_TE', '095_TE', 'PIV3']