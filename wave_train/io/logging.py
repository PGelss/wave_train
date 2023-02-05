import sys
from typing import Callable

def handle_logging(func: Callable):
    """
    Wrapper function that can be used to automatically log calls output to
    stdout in a logfile specified by the argument log_file. Argument must
    be explicitly provided. Logging type can be changed by providing the
    to_console argument in kwargs of func.

    func: Callable
        function with kwargs optionality
    

    log_file: str
        Name of the logfile  
    to_console: boolean
        If True, output will also be written to stdout, otherwise output
        will only be written to logfile

    Returns
    =======
        The result of func
    """
    def logging(*args, **kwargs):
        log_file    = ""
        to_console  = False

        if "log_file" in kwargs:
            log_file = kwargs["log_file"]

        if "to_console" in kwargs:
            to_console = kwargs["to_console"]
        else:
            to_console = True 

        if log_file and to_console:
            logger = TeeLogger(log_file)
            return func(*args, **kwargs)

        elif log_file and not to_console:
            logger = Logger(log_file)
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return logging

class TeeLogger:
    """
    Class for handling logging to a logfile while also writing to stdout.
    Previous value of stdout will be restored once TeeLogger is
    garbage collected.

    log_file: str
        Name of logfile
    """
    def __init__(self, log_file: str) -> None:
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
    """
    Class for handling logging to a logfile. All output to stdout
    will be redirected to the logfile.
    Previous value of stdout will be restored once Logger instance is
    garbage collected.

    log_file: str
        Name of logfile
    """
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