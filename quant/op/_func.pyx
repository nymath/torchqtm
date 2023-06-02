# distutils: language = c

import numpy as np
import pandas as pd
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memset

@cython.wraparound(False)
@cython.boundscheck(False)
def _group(x, int[::1] group, agg_func):
    if np.sum(np.isnan(x))+np.sum(np.isnan(group)):
        nan_mask = np.isnan(x) | np.isnan(group)

        x_copy = np.where(nan_mask, 0, x)
        group_copy = np.where(nan_mask, '0', group)

        labels, indices = np.unique(group_copy, return_inverse=True)
        grouped_val = np.array([agg_func(x_copy[group_copy == label]) for label in labels])
        rlt = grouped_val[indices]
        rlt = np.where(nan_mask, np.nan, rlt)
    else:
        labels, indices = np.unique(group, return_inverse=True)
        grouped_val = np.array([agg_func(x[group == label]) for label in labels])
        rlt = grouped_val[indices]
    return rlt

@cython.wraparound(False)
@cython.boundscheck(False)
def _group_neutralize_single(x, group):
    # Ensure x and group have same length
    assert len(x) == len(group), "Series x and group must have same length."

    # Calculate the mean of each group
    group_means = x.groupby(group).transform('mean')
    # Subtract the mean of each group from the corresponding elements in x
    neutralized_values = x - group.map(group_means)

    return neutralized_values