import sys
from typing import Callable

def handle_logging(func: Callable):
    def logging(*args, **kwargs):
        log_file    = ""
        teeing      = False

        if "log_file" in kwargs:
            log_file = kwargs["log_file"]

        if "split" in kwargs:
            teeing = kwargs["split"]
        else:
            teeing = True 

        if log_file and teeing:
            logger = TeeLogger(log_file)
            return func(*args, **kwargs)

        elif log_file and not teeing:
            logger = Logger(log_file)
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return logging

class TeeLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file
        self.stdout  = sys.stdout
        sys.stdout   = self
        
        self.file = open(self.log_file, "w")

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

class Logger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file 
        self.stdout  = sys.stdout
        sys.stdout   = self
        
        self.file = open(self.log_file, "w")

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)

    def flush(self):
        self.file.flush()