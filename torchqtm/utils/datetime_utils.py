import pandas as pd
import numpy as np
from torchqtm.constants import GLOBAL_DEFAULT_START


class DateTimeManager:
    def __init__(self, trading_calendar):
        self.trading_calendar = trading_calendar
        self.datetime: pd.Timestamp = GLOBAL_DEFAULT_START

    def set_dt(self, dt: pd.Timestamp):
        if isinstance(dt, pd.Timestamp):
            assert self.datetime <= dt, "Reverse backtest is not allowed"
            self.datetime = dt
        else:
            raise TypeError("dt should be a pd.Timestamp object")

    @property
    def current_dt(self):
        return self.datetime

    # def advance_time(self, delta):
    #     # delta should be a pd.DateOffset object
    #     if isinstance(delta, pd.DateOffset):
    #         self._current_time += delta
    #     else:
    #         raise TypeError("delta should be a pd.DateOffset object")


class DateTimeMixin:
    def __init__(self, dt_manager):
        self.datetime_manager = dt_manager

    @property
    def current_dt(self):
        return self.datetime_manager.current_dt

    def set_dt(self, dt):
        self.datetime_manager.set_dt(dt)

# TradingAlgorithm, Account, DataPortal


def convert_date_to_int(dt):
    t = dt.year * 10000 + dt.month * 100 + dt.day
    t *= 1e6
    return t


def convert_dates_to_int(dates: pd.DatetimeIndex):
    intDates = dates.year * 10000 + dates.month * 100 + dates.day
    intDates *= 1e6
    return intDates.values.astype(np.uint64)