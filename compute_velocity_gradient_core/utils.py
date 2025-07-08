import os, builtins, time, sys

def print(text, **kwargs):
    builtins.print(text, **kwargs)
    os.fsync(sys.stdout)

def setup_logging(log_file):
    sys.stdout = open(log_file, "w", buffering=1)

def init_logging_from_cut(cut,data_type):
    log_file = f"log_invariants_{cut}_{data_type}.txt"
    setup_logging(log_file)

def timer(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"The total compute time is: {int(time.time() - start)} s")
        return result
    return inner