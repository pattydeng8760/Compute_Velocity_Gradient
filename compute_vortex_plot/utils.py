import os
import sys
import time
import logging
from functools import wraps

def print(*args, **kwargs):
    """Custom print function that also logs to file."""
    # If logging is configured, use logging (which handles both file and console)
    if logging.getLogger().hasHandlers():
        message = ' '.join(str(arg) for arg in args)
        logging.info(message)
    else:
        # If no logging configured, use built-in print
        __builtins__['print'](*args, **kwargs)

def setup_logging(log_file):
    sys.stdout = open(log_file, "w", buffering=1)

def init_logging_from_cut(cut, data_type='LES'):
    """Initialize logging and redirect stdout to a log file."""
    log_filename = f"log_vortex_plot_{cut}_{data_type}.txt"
    
    # Delete existing log file if it exists to start fresh
    if os.path.exists(log_filename):
        os.remove(log_filename)
        # Use built-in print for this message before logging is set up
        __builtins__['print'](f"Deleted existing log file: {log_filename}")
    
    # Clear any existing logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    # Override the built-in print function globally
    import builtins
    builtins.print = print
    
    print(f"Logging initialized. Output will be saved to: {log_filename}")

def timer(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"The total compute time is: {int(time.time() - start)} s")
        return result
    return inner