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
        import builtins
        builtins.print(*args, **kwargs)

def init_logging_from_cut(cut, data_type='LES'):
    """Initialize logging and redirect stdout to a log file."""
    log_filename = f"log_vortex_detect_{cut}_{data_type}.txt"
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