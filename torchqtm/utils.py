# Time Operation
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time


class Timer(object):
    """
    >>> import numpy as np
    >>> with Timer("mytest.ipynb np.mean") as f:
    ...     x = np.random.normal(0, 1, (1000000, 100))
    ...     np.mean(x, axis=1)
    """
    def __init__(self, program: str = None):
        self.program = program
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, trace):
        print("{}: {:.4f}ms".format(self.program, 1000*(time.time() - self.start)))


