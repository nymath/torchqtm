import codecs
import json
import locale
import os
import sys
from copy import copy
from itertools import chain
from typing import Dict, Iterable, Optional, Literal

import h5py
import numpy as np
import pandas as pd
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod

from functools import lru_cache

from torchqtm.data.bundle.rqalpha import DEFAULT_PATH


TERM_ALLOWED = Literal['0S', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y',
                        '8Y', '9Y', '10Y', '15Y', '20Y', '30Y', '40Y', '50Y']


def convert_date_to_int(dt):
    t = dt.year * 10000 + dt.month * 100 + dt.day
    t *= 1e6
    return t


def convert_dates_to_int(dates: pd.DatetimeIndex):
    intDates = dates.year * 10000 + dates.month * 100 + dates.day
    intDates *= 1e6
    return intDates.values.astype(np.uint64)


class TradingCalendarReader(object):
    def __init__(self, path=None):
        if path is None:
            self.path = os.path.join(DEFAULT_PATH, "trading_dates.npy")
        else:
            self.path = path

    def get_trading_calendar(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.Series(np.load(self.path, allow_pickle=False), dtype=str))


# TODO: implement it
class FutureInfoReader(object):
    def __init__(self, path=None, custom_future_info=None):
        if path is None:
            self.path = os.path.join(DEFAULT_PATH, "future_info.json")
        else:
            self.path = path
        with open(self.path, "r") as json_file:
            self._default_data = {
                item.get("order_book_id") or item.get("underlying_symbol"): self._process_future_info_item(
                    item
                ) for item in json.load(json_file)
            }
        self._custom_data = custom_future_info
        self._future_info = {}

    @classmethod
    def _process_future_info_item(cls, item):
        item["commission_type"] = item["commission_type"]
        return item

    def get_future_info(self, instrument):
        order_book_id = instrument.order_book_id
        try:
            return self._future_info[order_book_id]
        except KeyError:
            custom_info = self._custom_data.get(order_book_id) or self._custom_data.get(instrument.underlying_symbol)
            info = self._default_data.get(order_book_id) or self._default_data.get(instrument.underlying_symbol)
            if custom_info:
                info = copy(info) or {}
                info.update(custom_info)
            elif not info:
                raise NotImplementedError(_("unsupported future instrument {}").format(order_book_id))
            return self._future_info.setdefault(order_book_id, info)


class InstrumentStore(object):
    def __init__(self, instruments, instrument_type):
        # type: (Iterable[Instrument], INSTRUMENT_TYPE) -> None
        self._instrument_type = instrument_type
        self._instruments = {}
        self._sym_id_map = {}

        for ins in instruments:
            if ins.type != instrument_type:
                continue
            self._instruments[ins.order_book_id] = ins
            self._sym_id_map[ins.symbol] = ins.order_book_id

    @property
    def instrument_type(self):
        # type: () -> INSTRUMENT_TYPE
        return self._instrument_type

    @property
    def all_id_and_syms(self):
        # type: () -> Iterable[str]
        return chain(self._instruments.keys(), self._sym_id_map.keys())

    def get_instruments(self, id_or_syms):
        # type: (Optional[Iterable[str]]) -> Iterable[Instrument]
        if id_or_syms is None:
            return self._instruments.values()
        order_book_ids = set()
        for i in id_or_syms:
            if i in self._instruments:
                order_book_ids.add(i)
            elif i in self._sym_id_map:
                order_book_ids.add(self._sym_id_map[i])
        return (self._instruments[i] for i in order_book_ids)


class ShareTransformationStore(object):
    def __init__(self, f):
        with codecs.open(f, 'r', encoding="utf-8") as store:
            self._share_transformation = json.load(store)

    def get_share_transformation(self, order_book_id):
        try:
            transformation_data = self._share_transformation[order_book_id]
        except KeyError:
            return
        return transformation_data["successor"], transformation_data["share_conversion_ratio"]


def _file_path(path):
    # why do this? non-ascii path in windows!!
    if sys.platform == "win32":
        try:
            l = locale.getlocale(locale.LC_ALL)[1]
        except TypeError:
            l = None
        if l and l.lower() == "utf-8":
            return path.encode("utf-8")
    return path


class h5_file:
    def __init__(self, path, *args, mode="r", **kwargs):
        self.path = path
        self.args = args
        self.mode = mode
        self.kwargs = kwargs

    def __enter__(self):
        try:
            self.h5 = h5py.File(self.path, *self.args, mode=self.mode, **self.kwargs)
        except OSError as e:
            raise RuntimeError(
                "open data bundle failed, you can remove {} and try to regenerate bundle: {}".format(self.path, e))
        return self.h5

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5.close()


class StockDailyBarReader(object):
    DEFAULT_DTYPES = np.dtype([
        ('datetime', np.uint64),
        ('open', np.float64),
        ('close', np.float64),
        ('high', np.float64),
        ('low', np.float64),
        ('volume', np.float64),
    ])

    ATTRIBUTES = ["datetime", "open", "close", "high", "low", "limit_up", "limit_down", "volume", "total_turnover"]

    ATTRIBUTE_TO_INDEX = {
        "datetime": 0,
        "open": 1,
        "close": 2,
        "high": 3,
        "low": 4,
        "volume": 5,
    }

    def __init__(self, path):
        if path is None:
            self._path = os.path.join(DEFAULT_PATH, "stocks.h5")
        else:
            self._path = path

    def load_data(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                return h5[symbol][:]
            except KeyError:
                return np.empty(0, dtype=self.DEFAULT_DTYPES)

    def get_date_range(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                data = h5[symbol]
                return data[0]['datetime'], data[-1]['datetime']
            except KeyError:
                return 20050104 * 1e6, 20050104 * 1e6

    def symbols(self):
        with h5_file(self._path) as h5:
            return list(h5.keys())


class FutureDailyBarReader(object):
    DEFAULT_DTYPES = np.dtype([
        ('datetime', np.uint64),
        ('open', np.float64),
        ('close', np.float64),
        ('high', np.float64),
        ('low', np.float64),
        ('volume', np.float64),
    ])

    def __init__(self, path=None):
        if path is None:
            self._path = os.path.join(DEFAULT_PATH, "futures.h5")
        else:
            self._path = path

    def load_data(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                return h5[symbol][:]
            except KeyError:
                return np.empty(0, dtype=self.DEFAULT_DTYPES)

    def get_date_range(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                data = h5[symbol]
                return data[0]['datetime'], data[-1]['datetime']
            except KeyError:
                return 20050104 * 1e6, 20050104 * 1e6

    def symbols(self):
        with h5_file(self._path) as h5:
            return list(h5.keys())


class FundDailyBarReader(object):
    DEFAULT_DTYPES = np.dtype([
        ('datetime', np.uint64),
        ('open', np.float64),
        ('close', np.float64),
        ('high', np.float64),
        ('low', np.float64),
        ('volume', np.float64),
    ])

    def __init__(self, path=None):
        if path is None:
            self._path = os.path.join(DEFAULT_PATH, "funds.h5")
        else:
            self._path = path

    def load_data(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                return h5[symbol][:]
            except KeyError:
                return np.empty(0, dtype=self.DEFAULT_DTYPES)

    def get_date_range(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                data = h5[symbol]
                return data[0]['datetime'], data[-1]['datetime']
            except KeyError:
                return 20050104 * 1e6, 20050104 * 1e6

    def symbols(self):
        with h5_file(self._path) as h5:
            return list(h5.keys())


class IndexDailyBarReader(object):
    DEFAULT_DTYPES = np.dtype([
        ('datetime', np.uint64),
        ('open', np.float64),
        ('close', np.float64),
        ('high', np.float64),
        ('low', np.float64),
        ('volume', np.float64),
    ])

    def __init__(self, path=None):
        if path is None:
            self._path = os.path.join(DEFAULT_PATH, "indexes.h5")
        else:
            self._path = path

    def load_data(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                return h5[symbol][:]
            except KeyError:
                return np.empty(0, dtype=self.DEFAULT_DTYPES)

    def get_date_range(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                data = h5[symbol]
                return data[0]['datetime'], data[-1]['datetime']
            except KeyError:
                return 20050104 * 1e6, 20050104 * 1e6

    def symbols(self):
        with h5_file(self._path) as h5:
            return list(h5.keys())


class DividendsReader(object):
    DEFAULT_DTYPES = np.dtype([
        ('book_closure_date', np.uint64),
        ('announcement_date', np.float64),
        ('dividend_cash_before_tax', np.float64),
        ('ex_dividend_date', np.uint64),
        ('payable_date', np.uint64),
        ('round_lot', np.float64)
    ])

    def __init__(self, path=None):
        if path is None:
            self._path = os.path.join(DEFAULT_PATH, "dividends.h5")
        else:
            self._path = path

    def load_data(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                return h5[symbol][:]
            except KeyError:
                return None


class SplitsReader(object):
    DEFAULT_DTYPES = np.dtype([
        ("ex_date", np.uint64),
        ("split_factor", np.float64),
    ])

    def __init__(self, path=None):
        if path is None:
            self._path = os.path.join(DEFAULT_PATH, "split_factor.h5")
        else:
            self._path = path

    def load_data(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                return h5[symbol][:]
            except KeyError:
                return None


class YieldCurveReader(object):
    def __init__(self, path=None):
        if path is None:
            self._path = os.path.join(DEFAULT_PATH, "yield_curve.h5")
        else:
            self._path = path
        self._data = None

    def load_data(
            self,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp,
            term: Optional[TERM_ALLOWED] = None,
    ):
        with h5_file(self._path) as h5:
            self._data = h5["data"][:]

        d1 = int(convert_date_to_int(start_date) / 1e6)
        d2 = int(convert_date_to_int(end_date) / 1e6)

        s = self._data['date'].searchsorted(d1)
        e = self._data['date'].searchsorted(d2, side='right')

        if e == len(self._data):
            e -= 1
        if self._data[e]['date'] == d2:
            e += 1

        if e < s:
            return None

        df = pd.DataFrame(self._data[s:e])
        df.index = pd.to_datetime([str(d) for d in df['date']])
        del df['date']

        if term is not None:
            return df[term]
        return df


class ExCumFactorReader(object):
    def __init__(self, path=None):
        if path is None:
            self._path = os.path.join(DEFAULT_PATH, "ex_cum_factor.h5")
        else:
            self._path = path

    @lru_cache(None)
    def load_data(self, symbol: str):
        with h5_file(self._path) as h5:
            try:
                return h5[symbol][:]
            except KeyError:
                return None


class DatesReader(object):
    def __init__(self, path=None):
        if path is None:
            self._path = os.path.join(DEFAULT_PATH, "trading_dates.npy")
        else:
            self._path = path

    @lru_cache(None)
    def get_days(self, order_book_id):
        with h5_file(self._f) as h5:
            try:
                days = h5[order_book_id][:]
            except KeyError:
                return set()
            return set(days.tolist())

    def contains(self, symbol, dates):
        date_set = self.get_days(symbol)
        if not date_set:
            return None

        def _to_dt_int(d):
            if isinstance(d, (int, np.int64, np.uint64)):
                return int(d // 1000000) if d > 100000000 else int(d)
            else:
                return d.year * 10000 + d.month * 100 + d.day

        return [(_to_dt_int(d) in date_set) for d in dates]


if __name__ == "__main__":
    bar_stock = StockDailyBarReader(os.path.join(DEFAULT_PATH, 'stocks.h5'))
    bar_future = FutureDailyBarReader(os.path.join(DEFAULT_PATH, 'futures.h5'))
    split_reader = SplitsReader()
    calender = TradingCalendarReader()
    calender.get_trading_calendar()
    bar_stock.load_data("000001.XSHE")
    yield_curve = YieldCurveReader()
    yield_curve.load_data(pd.Timestamp('20100101'), pd.Timestamp('20150101'))
    bar_stock.get_date_range("000001.XSHE")
    bar_future.load_data("IF2101")

    # with h5_file(os.path.join(DEFAULT_PATH, 'futures.h5'), mode='r') as f:
    #     f["000001.XSHE"][:]


