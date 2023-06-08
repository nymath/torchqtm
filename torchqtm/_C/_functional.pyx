# cython: language_level=3
# https://cython.readthedocs.io/en/stable/src/userguide/numpy_tutorial.html
cimport numpy
import numpy as np
cimport cython
import pandas as pd
from torchqtm.config import __OP_MODE__
from sklearn.linear_model import LinearRegression

ctypedef fused ArrayType:
    float
    double
    long long



@cython.wraparound(False)
@cython.boundscheck(False)
def _regression_neut(Y, X):
    result = []
    for i in range(len(Y)):
        y = Y.values[i]
        y_demean = y - np.nanmean(y)
        x = X.values[i]
        x_demean = x - np.nanmean(x)
        residuals = y_demean - (np.nanmean(y_demean * x_demean) / np.nanvar(x_demean)) * x_demean
        result.append(residuals)
    return pd.DataFrame(np.array(result), index=Y.index, columns=Y.columns)


@cython.wraparound(False)
@cython.boundscheck(False)
def _regression_neuts(Y, others):
    result = []
    for i in range(len(Y)):
        y = Y.values[i]
        X = np.concatenate([x.values[i].reshape(-1, 1) for x in others], axis=1)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        result.append(residuals)
    return pd.DataFrame(np.array(result), index=Y.index, columns=Y.columns)


@cython.wraparound(False)
@cython.boundscheck(False)
def regression_neut(Y, others):
    assert isinstance(Y, pd.DataFrame)
    if not isinstance(others, list):
        return _regression_neut(Y, others)
    else:
        if __OP_MODE__ == "STABLE":
            return _regression_neuts(Y, others)
        else:
            pass

# roll_apply.pyx
cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def roll_apply(np.ndarray[double, ndim=1] values, np.ndarray[int, ndim=1] begin, np.ndarray[int, ndim=1] end,
               int minimum_periods, func):
    cdef Py_ssize_t i, start, stop, count_nan
    cdef np.ndarray[double] result = np.empty(len(begin))

    with nogil:
        for i in cython.parallel.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window))
            if len(window) - count_nan >= minimum_periods:
                result[i] = func(window)
            else:
                result[i] = np.nan
    return result
