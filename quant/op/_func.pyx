# distutils: language = c

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset

def SMA(np.ndarray[np.float64_t, ndim=1] input, int period):
    cdef int input_size = input.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] output = np.empty(input_size)
    cdef double sum = 0.0
    cdef int i

    for i in range(input_size):
        sum += input[i]
        if i >= period:
            sum -= input[i - period]
        if i >= period - 1:
            output[i - period + 1] = sum / period
    return output


