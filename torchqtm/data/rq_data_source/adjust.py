import pandas as pd
import numpy as np
from bisect import bisect_right

import typing


def convert_date_to_int(dt):
    t = dt.year * 10000 + dt.month * 100 + dt.day
    t *= 1000000
    return t


def convert_dates_to_int(dates: pd.DatetimeIndex):
    intDates = dates.year * 10000 + dates.month * 100 + dates.day
    intDates *= 10e6
    return intDates.values.astype(np.uint64)


PRICE_FIELDS = {
    'open', 'close', 'high', 'low', 'limit_up', 'limit_down', 'acc_net_value', 'unit_net_value'
}

FIELDS_REQUIRE_ADJUSTMENT = set(list(PRICE_FIELDS) + ['volume'])


def _factor_for_date(dates, factors, d):
    pos = bisect_right(dates, d)
    return factors[pos-1]

# array([(             0,   1.     ), (19910502000000,   1.40968),
#        (19910817000000,   1.63169), (19920323000000,   2.46272),
#        (19930524000000,   4.6701 ), (19940711000000,   7.47216),
#        (19950925000000,   9.24767), (19960527000000,  18.4953 ),
#        (19970825000000,  27.9199 ), (19991018000000,  28.6789 ),
#        (20001106000000,  22.9762 ), (20020723000000,  23.2163 ),
#        (20030929000000,  23.5953 ), (20070620000000,  25.9629 ),
#        (20081031000000,  33.8521 ), (20121019000000,  34.1045 ),
#        (20130620000000,  55.0537 ), (20140612000000,  66.9741 ),
#        (20150413000000,  81.0815 ), (20160616000000,  98.7449 ),
#        (20170721000000, 100.188  ), (20180712000000, 101.764  ),
#        (20190626000000, 102.875  ), (20200528000000, 104.629  ),
#        (20210514000000, 105.452  ), (20220722000000, 107.333  )],
#       dtype=[('start_date', '<i8'), ('ex_cum_factor', '<f8')])


def adjust_bars(
        bars: np.ndarray,
        ex_factors: np.ndarray,
        fields: typing.Iterable[str],
        adjust_type: str,
        adjust_origin: pd.Timestamp,
):
    """
    Parameters
    ----------
    bars : np.ndarray
    ex_factors : np.ndarray
    fields : 不太确定
    adjust_type : str
    adjust_origin: pd.Timestamp
    """

    if ex_factors is None or len(bars) == 0:
        return bars

    dates = ex_factors['start_date']
    ex_cum_factors = ex_factors['ex_cum_factor']

    if adjust_type == 'pre':
        adjust_orig_dt = np.int64(convert_date_to_int(adjust_origin))
        base_adjust_rate = _factor_for_date(dates, ex_cum_factors, adjust_orig_dt)
    else:
        base_adjust_rate = 1.0

    start_date = bars['datetime'][0]
    end_date = bars['datetime'][-1]

    if (_factor_for_date(dates, ex_cum_factors, start_date) == base_adjust_rate and
            _factor_for_date(dates, ex_cum_factors, end_date) == base_adjust_rate):
        return bars

    factors = ex_cum_factors.take(dates.searchsorted(bars['datetime'], side='right') - 1)

    # 复权
    bars = np.copy(bars)
    factors /= base_adjust_rate
    if isinstance(fields, str):
        if fields in PRICE_FIELDS:
            bars[fields] *= factors
            return bars
        elif fields == 'volume':
            bars[fields] *= (1 / factors)
            return bars
        # should not got here
        return bars

    for f in bars.dtype.names:
        if f in PRICE_FIELDS:
            bars[f] *= factors
        elif f == 'volume':
            bars[f] *= (1 / factors)
    return bars
