# distutils: language=c++


cimport cython
# from cython cimport PyObject

cimport numpy as cnp
import numpy as np
import pandas as pd
from torchqtm.finance.positions.position import Position
from torchqtm.assets import Future

cdef class PositionStats:
    """Compute values from the current positions

    Attributes
    ----------
    """
    cdef cnp.float64_t gross_exposure
    cdef cnp.float64_t gross_value
    cdef cnp.float64_t long_exposure
    cdef cnp.float64_t long_value
    cdef cnp.float64_t net_exposure
    cdef cnp.float64_t net_value
    cdef cnp.float64_t short_exposure
    cdef cnp.float64_t short_value
    cdef cnp.uint64_t longs_count
    cdef cnp.uint64_t shorts_count
    cdef object position_exposure_array
    cdef object position_exposure_series

    cdef object underlying_value_array
    cdef object underlying_index_array

    @classmethod
    def new(cls):
        cdef PositionStats self = cls()
        self.position_exposure_series = es = pd.Series(
            np.array([], dtype='float64'),
            index=np.array([], dtype='int64'),
        )
        self.underlying_value_array = self.position_exposure_array = es.values
        self.underlying_index_array = es.index.values
        return self


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_positions_tracker_stats(positions, PositionStats stats):
    """Calculate various stats about the current positions
    
    Parameters
    ----------
    positions : OrderedDict
    stats : PositionStats
    
    Returns
    -------
    position_stats : PositionStats
        The computed statistics
    """
    cdef Py_ssize_t n_pos = len(positions)
    cdef cnp.ndarray[cnp.uint64_t] index
    cdef cnp.ndarray[cnp.float64_t] position_exposure

    cdef cnp.ndarray[cnp.int64_t] old_index = stats.underlying_index_array
    cdef cnp.ndarray[cnp.float64_t] old_position_exposure = stats.underlying_index_array

    cdef cnp.float64_t value
    cdef cnp.float64_t exposure

    cdef cnp.float64_t net_value
    cdef cnp.float64_t gross_value
    cdef cnp.float64_t long_value = 0.0
    cdef cnp.float64_t short_value = 0.0

    cdef cnp.float64_t net_exposure
    cdef cnp.float64_t gross_exposure
    cdef cnp.float64_t long_exposure = 0.0
    cdef cnp.float64_t short_exposure = 0.0

    cdef cnp.uint64_t longs_count = 0
    cdef cnp.uint64_t shorts_count = 0


    # if len(old_index) < n_pos:
    #     # we don't have enough space in the cached buffer, allocate a new
    #     # array
    #     stats.underlying_index_array = index = np.empty(n_pos, dtype='int64')
    #     stats.underlying_value_array = position_exposure = np.empty(
    #         n_pos,
    #         dtype='float64',
    #     )
    #
    #     stats.position_exposure_array = position_exposure
    #     # create a new series to expose the arrays
    #     stats.position_exposure_series = pd.Series(
    #         position_exposure,
    #         index=index,
    #     )
    # elif len(old_index) > n_pos:
    #     # we have more space than needed, slice off the extra but leave it
    #     # available
    #     index = old_index[:n_pos]
    #     position_exposure = old_position_exposure[:n_pos]
    #
    #     stats.position_exposure_array = position_exposure
    #     # create a new series with the sliced arrays
    #     stats.position_exposure_series = pd.Series(
    #         position_exposure,
    #         index=index,
    #     )
    # else:
    #     # we have exactly the right amount of space, no slicing or allocation
    #     # needed
    #     index = old_index
    #     position_exposure = old_position_exposure
    #
    #     stats.position_exposure_array = position_exposure
    #     stats.position_exposure_series = pd.Series(
    #         position_exposure,
    #         index=index,
    #     )

    # cdef InnerPosition position
    cdef Py_ssize_t ix = 0

    for position in positions.values():
        # NOTE: this loop does a lot of stuff!
        # we call this function every time the portfolio value is needed,
        # which is at least once per simulation day, so let's not iterate
        # through every single position multiple times.
        exposure = position.amount * position.last_sale_price

        if type(position.asset) is Future:
            # Futures don't have an inherent position value.
            value = 0

            # unchecked cast, this is safe because we do a type check above
            exposure *= position.asset.price_multiplier
        else:
            value = exposure

        if exposure > 0:
            longs_count += 1
            long_value += value
            long_exposure += exposure
        elif exposure < 0:
            shorts_count += 1
            short_value += value
            short_exposure += exposure

        # index[ix] = position.asset.sid
        # position_exposure[ix] = exposure
        # ix += 1

    net_value = long_value + short_value
    gross_value = long_value - short_value

    net_exposure = long_exposure + short_exposure
    gross_exposure = long_exposure - short_exposure

    stats.gross_exposure = gross_exposure
    stats.gross_value = gross_value
    stats.long_exposure = long_exposure
    stats.long_value = long_value
    stats.longs_count = longs_count
    stats.net_exposure = net_exposure
    stats.net_value = net_value
    stats.short_exposure = short_exposure
    stats.short_value = short_value
    stats.shorts_count = shorts_count
