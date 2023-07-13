# distutils: language=c++
from typing import Type

import numpy as np
cimport numpy as cnp
import cython
# from Cython.Includes.numpy import ndarray
from libc.math cimport round
from libcpp.deque cimport deque
from numpy cimport (
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

from pandas._libs.algos cimport TiebreakEnumType

# from torchqtm._libs.window.aggregations import skiplist_init

# All numeric types except complex
ctypedef fused numeric_t:
    int8_t
    int16_t
    int32_t
    int64_t

    uint8_t
    uint16_t
    uint32_t
    uint64_t

    float32_t
    float64_t


from numpy cimport (
    float32_t,
    float64_t,
    int64_t,
    ndarray,
)

cdef:
    float32_t MINfloat32 = np.NINF
    float64_t MINfloat64 = np.NINF

    float32_t MAXfloat32 = np.inf
    float64_t MAXfloat64 = np.inf

    float64_t NaN = <float64_t>np.NaN

ctypedef struct node_t:
    node_t ** next
    int *width
    double value
    int is_nil
    int levels
    int ref_count

ctypedef struct skiplist_t:
    node_t *head
    node_t ** tmp_chain
    int *tmp_steps
    int size
    int maxlevels


cdef bint isnan(float64_t x) nogil:
    return x != x

cdef bint notnan(float64_t x) nogil:
    return x == x

cdef bint signbit(float64_t x) nogil:
    if x < 0.0:
        return 1
    else:
        return 0


#
#
cdef inline numeric_t calc_mm(int64_t minp, Py_ssize_t nobs,
                              numeric_t value) nogil:
    cdef:
        numeric_t result

    if numeric_t in cython.floating:
        if nobs >= minp:
            result = value
        else:
            result = NaN
    else:
        result = value

    return result


cdef inline numeric_t init_mm(numeric_t ai, Py_ssize_t *nobs, bint is_max) nogil:

    if numeric_t in cython.floating:
        if ai == ai:
            nobs[0] = nobs[0] + 1
        elif is_max:
            if numeric_t == cython.float:
                ai = MINfloat32
            else:
                ai = MINfloat64
        else:
            if numeric_t == cython.float:
                ai = MAXfloat32
            else:
                ai = MAXfloat64

    else:
        nobs[0] = nobs[0] + 1

    return ai

cdef inline void remove_mm(numeric_t aold, Py_ssize_t *nobs) nogil:
    """ remove a value from the mm calc """
    if numeric_t in cython.floating and aold == aold:
        nobs[0] = nobs[0] - 1


cdef inline float64_t calc_sum(int64_t minp, int64_t nobs, float64_t sum_x) nogil:
    cdef:
        float64_t result

    if nobs == 0 == minp:
        result = 0
    elif nobs >= minp:
        result = sum_x
    else:
        result = NaN

    return result


cdef inline void add_sum(float64_t val, int64_t *nobs, float64_t *sum_x,
                         float64_t *compensation) nogil:
    """ add a value from the sum calc using Kahan summation """

    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] + 1
        y = val - compensation[0]
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t


cdef inline void remove_sum(float64_t val, int64_t *nobs, float64_t *sum_x,
                            float64_t *compensation) nogil:
    """ remove a value from the sum calc using Kahan summation """

    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] - 1
        y = - val - compensation[0]
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t


def roll_sum(const float64_t[:] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        Py_ssize_t i, j
        float64_t sum_x, compensation_add, compensation_remove
        int64_t s, e
        int64_t nobs = 0, N = len(start)
        ndarray[float64_t] output
        bint is_monotonic_increasing_bounds

    is_monotonic_increasing_bounds = True

    output = np.empty(N, dtype=np.float64)

    with nogil:

        for i in range(0, N):
            s = start[i]
            e = end[i]

            if i == 0 or not is_monotonic_increasing_bounds or s >= end[i - 1]:

                # setup

                sum_x = compensation_add = compensation_remove = 0
                nobs = 0
                for j in range(s, e):
                    add_sum(values[j], &nobs, &sum_x, &compensation_add)

            else:

                # calculate deletes
                for j in range(start[i - 1], s):
                    remove_sum(values[j], &nobs, &sum_x, &compensation_remove)

                # calculate adds
                for j in range(end[i - 1], e):
                    add_sum(values[j], &nobs, &sum_x, &compensation_add)

            output[i] = calc_sum(minp, nobs, sum_x)

            if not is_monotonic_increasing_bounds:
                nobs = 0
                sum_x = 0.0
                compensation_remove = 0.0

    return output


def roll_max(ndarray[float64_t] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    """
    Moving max of 1d array of any numeric type along axis=0 ignoring NaNs.

    Parameters
    ----------
    values : np.ndarray[np.float64]
    window : int, size of rolling window
    minp : if number of observations in window
          is below this, output a NaN
    index : ndarray, optional
       index for window computation
    closed : 'right', 'left', 'both', 'neither'
            make the interval closed on the right, left,
            both or neither endpoints

    Returns
    -------
    np.ndarray[float]
    """
    return _roll_min_max(values, start, end, minp, is_max=1)


def roll_min(ndarray[float64_t] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    """
    Moving min of 1d array of any numeric type along axis=0 ignoring NaNs.

    Parameters
    ----------
    values : np.ndarray[np.float64]
    window : int, size of rolling window
    minp : if number of observations in window
          is below this, output a NaN
    index : ndarray, optional
       index for window computation

    Returns
    -------
    np.ndarray[float]
    """
    return _roll_min_max(values, start, end, minp, is_max=0)


cdef _roll_min_max(ndarray[numeric_t] values,
                   ndarray[int64_t] starti,
                   ndarray[int64_t] endi,
                   int64_t minp,
                   bint is_max):
    cdef:
        numeric_t ai
        int64_t curr_win_size, start
        Py_ssize_t i, k, nobs = 0, N = len(starti)
        deque Q[int64_t]  # min/max always the front
        deque W[int64_t]  # track the whole window for nobs compute
        ndarray[float64_t, ndim=1] output

    output = np.empty(N, dtype=np.float64)
    Q = deque[int64_t]()
    W = deque[int64_t]()

    with nogil:

        # This is using a modified version of the C++ code in this
        # SO post: https://stackoverflow.com/a/12239580
        # The original impl didn't deal with variable window sizes
        # So the code was optimized for that

        # first window's size
        curr_win_size = endi[0] - starti[0]
        # GH 32865
        # Anchor output index to values index to provide custom
        # BaseIndexer support
        for i in range(N):

            curr_win_size = endi[i] - starti[i]
            if i == 0:
                start = starti[i]
            else:
                start = endi[i - 1]

            for k in range(start, endi[i]):
                ai = init_mm(values[k], &nobs, is_max)
                # Discard previous entries if we find new min or max
                if is_max:
                    while not Q.empty() and ((ai >= values[Q.back()]) or
                                             values[Q.back()] != values[Q.back()]):
                        Q.pop_back()
                else:
                    while not Q.empty() and ((ai <= values[Q.back()]) or
                                             values[Q.back()] != values[Q.back()]):
                        Q.pop_back()
                Q.push_back(k)
                W.push_back(k)

            # Discard entries outside and left of current window
            while not Q.empty() and Q.front() <= starti[i] - 1:
                Q.pop_front()
            while not W.empty() and W.front() <= starti[i] - 1:
                remove_mm(values[W.front()], &nobs)
                W.pop_front()

            # Save output based on index in input value array
            if not Q.empty() and curr_win_size > 0:
                output[i] = calc_mm(minp, nobs, values[Q.front()])
            else:
                output[i] = NaN

    return output


cdef inline float64_t calc_mean(int64_t minp, Py_ssize_t nobs,
                                Py_ssize_t neg_ct, float64_t sum_x) nogil:
    cdef:
        float64_t result

    if nobs >= minp and nobs > 0:
        result = sum_x / <float64_t>nobs
        if neg_ct == 0 and result < 0:
            # all positive
            result = 0
        elif neg_ct == nobs and result > 0:
            # all negative
            result = 0
        else:
            pass
    else:
        result = NaN
    return result


cdef inline void add_mean(float64_t val, Py_ssize_t *nobs, float64_t *sum_x,
                          Py_ssize_t *neg_ct, float64_t *compensation) nogil:
    """ add a value from the mean calc using Kahan summation """
    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] + 1
        y = val - compensation[0]
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t
        if signbit(val):
            neg_ct[0] = neg_ct[0] + 1


cdef inline void remove_mean(float64_t val, Py_ssize_t *nobs, float64_t *sum_x,
                             Py_ssize_t *neg_ct, float64_t *compensation) nogil:
    """ remove a value from the mean calc using Kahan summation """
    cdef:
        float64_t y, t

    if notnan(val):
        nobs[0] = nobs[0] - 1
        y = - val - compensation[0]
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t
        if signbit(val):
            neg_ct[0] = neg_ct[0] - 1


@cython.boundscheck(False)
@cython.wraparound(False)
def roll_mean(const float64_t[:] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        float64_t val, compensation_add, compensation_remove, sum_x
        int64_t s, e
        Py_ssize_t nobs, i, j, neg_ct, N = len(start)
        ndarray[float64_t] output
        bint is_monotonic_increasing_bounds

    is_monotonic_increasing_bounds = True

    output = np.empty(N, dtype=np.float64)

    with nogil:

        for i in range(0, N):
            s = start[i]
            e = end[i]

            if i == 0 or not is_monotonic_increasing_bounds or s >= end[i - 1]:

                compensation_add = compensation_remove = sum_x = 0
                nobs = neg_ct = 0
                # setup
                for j in range(s, e):
                    val = values[j]
                    add_mean(val, &nobs, &sum_x, &neg_ct, &compensation_add)

            else:

                # calculate deletes
                for j in range(start[i - 1], s):
                    val = values[j]
                    remove_mean(val, &nobs, &sum_x, &neg_ct, &compensation_remove)

                # calculate adds
                for j in range(end[i - 1], e):
                    val = values[j]
                    add_mean(val, &nobs, &sum_x, &neg_ct, &compensation_add)

            output[i] = calc_mean(minp, nobs, neg_ct, sum_x)

            if not is_monotonic_increasing_bounds:
                nobs = 0
                neg_ct = 0
                sum_x = 0.0
                compensation_remove = 0.0
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def roll_apply_1D(object obj,
                  ndarray[int64_t] start,
                  ndarray[int64_t] end,
                  int64_t minp,
                  object function,
                  tuple args,
                  dict kwargs) -> np.ndarray:
    cdef:
        ndarray[float64_t] output, counts
        ndarray[float64_t, cast=True] arr
        Py_ssize_t i, s, e,
        Py_ssize_t N = len(start)
        Py_ssize_t n = len(obj)

    if n == 0:
        return np.array([], dtype=np.float64)

    arr = np.asarray(obj)

    # ndarray input
    if not arr.flags.c_contiguous:
        arr = arr.copy('C')

    counts = roll_sum(np.isfinite(arr).astype(float), start, end, minp)

    output = np.empty(N, dtype=np.float64)

    for i in range(N):

        s = start[i]
        e = end[i]

        if counts[i] >= minp:
                output[i] = function(arr[s:e], *args, **kwargs)
        else:
            output[i] = NaN

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def roll_apply_2D(object obj,
                  ndarray[int64_t] start,
                  ndarray[int64_t] end,
                  int64_t minp,
                  object function,
                  tuple args,
                  dict kwargs) -> np.ndarray:
    cdef:
        ndarray[float64_t, ndim=2] output
        ndarray[float64_t, ndim=1] counts
        ndarray[float64_t, ndim=2, cast=True] arr
        Py_ssize_t i, s, e,
        Py_ssize_t N = len(start)
        Py_ssize_t n = len(obj)
        Py_ssize_t n_col = obj.shape[1]

    # if n == 0:
    #     return np.array([], dtype=np.float64)

    arr = np.asarray(obj)

    # ndarray input
    if not arr.flags.c_contiguous:
        arr = arr.copy('C')

    counts = roll_sum(np.isfinite(arr[:, 0]).astype(float), start, end, minp)

    output = np.empty((N, n_col), dtype=np.float64)

    for i in range(N):

        s = start[i]
        e = end[i]

        if counts[i] >= minp:
            output[i] = function(arr[s:e], *args, **kwargs)
        else:
            output[i] = NaN

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def roll_apply_3D(object obj,
                  ndarray[int64_t] start,
                  ndarray[int64_t] end,
                  int64_t minp,
                  object function,
                  tuple args,
                  dict kwargs) -> Type[ndarray]:
    cdef:
        ndarray[float64_t, ndim=2] output
        ndarray[float64_t, ndim=1] counts
        ndarray[float64_t, ndim=3, cast=True] arr
        Py_ssize_t i, s, e,
        Py_ssize_t N = len(start)
        Py_ssize_t n = len(obj)
        Py_ssize_t n_stocks = obj.shape[1]
        Py_ssize_t n_features = obj.shape[2]
    # if n == 0:
    #     return np.array([], dtype=np.float64)

    arr = np.asarray(obj)
    # ndarray input
    if not arr.flags.c_contiguous:
        arr = arr.copy('C')

    counts = roll_sum(np.isfinite(arr[:, 0, 0]).astype(float), start, end, minp)
    output = np.empty((N, n_stocks), dtype=np.float64)
    for i in range(N):
        s = start[i]
        e = end[i]

        if counts[i] >= minp:
            output[i] = function(arr[s:e], *args, **kwargs)
        else:
            output[i] = NaN
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def roll_weighted_sum(
    const float64_t[:] values, const float64_t[:] weights, int minp
) -> np.ndaray:
    return _roll_weighted_sum_mean(values, weights, minp, avg=0)


@cython.boundscheck(False)
@cython.wraparound(False)
def roll_weighted_mean(
    const float64_t[:] values, const float64_t[:] weights, int minp
) -> np.ndaray:
    return _roll_weighted_sum_mean(values, weights, minp, avg=1)


cdef float64_t[:] _roll_weighted_sum_mean(const float64_t[:] values,
                                          const float64_t[:] weights,
                                          int minp, bint avg):
    """
    Assume len(weights) << len(values)
    """
    cdef:
        float64_t[:] output, tot_wgt, counts
        Py_ssize_t in_i, win_i, win_n, in_n
        float64_t val_in, val_win, c, w

    in_n = len(values)
    win_n = len(weights)

    output = np.zeros(in_n, dtype=np.float64)
    counts = np.zeros(in_n, dtype=np.float64)
    if avg:
        tot_wgt = np.zeros(in_n, dtype=np.float64)

    elif minp > in_n:
        minp = in_n + 1

    minp = max(minp, 1)

    with nogil:
        if avg:
            for win_i in range(win_n):
                val_win = weights[win_i]
                if val_win != val_win:
                    continue

                for in_i in range(in_n - (win_n - win_i) + 1):
                    val_in = values[in_i]
                    if val_in == val_in:
                        output[in_i + (win_n - win_i) - 1] += val_in * val_win
                        counts[in_i + (win_n - win_i) - 1] += 1
                        tot_wgt[in_i + (win_n - win_i) - 1] += val_win

            for in_i in range(in_n):
                c = counts[in_i]
                if c < minp:
                    output[in_i] = NaN
                else:
                    w = tot_wgt[in_i]
                    if w == 0:
                        output[in_i] = NaN
                    else:
                        output[in_i] /= tot_wgt[in_i]

        else:
            for win_i in range(win_n):
                val_win = weights[win_i]
                if val_win != val_win:
                    continue

                for in_i in range(in_n - (win_n - win_i) + 1):
                    val_in = values[in_i]

                    if val_in == val_in:
                        output[in_i + (win_n - win_i) - 1] += val_in * val_win
                        counts[in_i + (win_n - win_i) - 1] += 1

            for in_i in range(in_n):
                c = counts[in_i]
                if c < minp:
                    output[in_i] = NaN

    return output