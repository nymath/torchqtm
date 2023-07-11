from collections import OrderedDict

import pandas as pd
import datetime
from pandas_market_calendars import get_calendar as calendar_template
import typing
import numpy as np

from torchqtm.constants import (
    GLOBAL_DEFAULT_START,
    GLOBAL_DEFAULT_END,
    GLOBAL_DEFAULT_MARKET_OPEN,
    GLOBAL_DEFAULT_MARKET_CLOSE,
    GLOBAL_DEFAULT_BEFORE_TRADING_START_OFFSET,
)
from torchqtm.types import DateLike

NANOSECONDS_PER_MINUTE = int(6e10)

from torchqtm.utils.datetime_utils import (
    convert_date_to_int,
    convert_dates_to_int,
)

# TODO: add the abc for ExchangeCalendar


class TradingDatesMixin(object):
    def __init__(self, trading_calendars):
        pass


class XSHGExchangeCalendar(object):

    calendar = calendar_template('SSE')
    market_open_time = datetime.time(hour=9, minute=30)
    market_break_start_time = datetime.time(hour=11, minute=30)
    market_break_end_time = datetime.time(hour=13, minute=0)
    market_close_time = datetime.time(hour=15, minute=0)

    @classmethod
    def default_start(cls) -> pd.Timestamp:
        return GLOBAL_DEFAULT_START

    @classmethod
    def default_end(cls) -> pd.Timestamp:
        return GLOBAL_DEFAULT_END

    def __init__(self,
                 start_date: DateLike = None,
                 end_date: DateLike = None,
                 side: typing.Literal["left", "right", "both", "neither"] = "left"):
        self.side = side
        if start_date is None:
            start_date = self.default_start()

        if end_date is None:
            end_date = self.default_end()

        if start_date >= end_date:
            raise ValueError(
                "`start` must be earlier than `end` although `start` parsed as"
                f" '{start_date}' and `end` as '{end_date}'."
            )
        self.start_date = start_date
        self.end_date = end_date
        self.valid_days: pd.DatetimeIndex = pd.DatetimeIndex(self.calendar.valid_days(start_date, end_date, tz=None))
        self.all_sessions = self.valid_days

        # numeric-friendly time
        # we can perform binary search on the calendar by introducing this
        self.numerical_valid_days: np.ndarray = convert_dates_to_int(self.valid_days)

        # format-friendly time
        self.market_opens: pd.DatetimeIndex = self.valid_days + pd.Timedelta(hours=self.market_open_time.hour,
                                                                             minutes=self.market_open_time.minute)
        self.market_break_starts: pd.DatetimeIndex = self.valid_days + pd.Timedelta(
            hours=self.market_break_start_time.hour,
            minutes=self.market_break_start_time.minute)
        self.market_break_ends: pd.DatetimeIndex = self.valid_days + pd.Timedelta(hours=self.market_break_end_time.hour,
                                                                                  minutes=self.market_break_end_time.minute)
        self.market_closes: pd.DatetimeIndex = self.valid_days + pd.Timedelta(hours=self.market_close_time.hour,
                                                                              minutes=self.market_close_time.minute)

        # Unix-based time
        self.opens_nanos = self.market_opens.values.astype(np.int64)
        self.break_starts_nanos = self.market_break_starts.values.astype(np.int64)
        self.break_ends_nanos = self.market_break_ends.values.astype(np.int64)
        self.closes_nanos = self.market_closes.values.astype(np.int64)

        # references
        self.schedule = pd.DataFrame(
            index=self.all_sessions,
            data=OrderedDict([
                ("market_open", self.market_opens),
                ("break_start", self.market_break_starts),
                ("break_end", self.market_break_ends),
                ("market_close", self.market_closes)
            ]),
            dtype='datetime64[ns]',
        )

    @property
    def name(self):
        return "上海证券交易所"

    @property
    def sessions(self):
        return self.valid_days

    @property
    def sessions_nanos(self):
        return self.valid_days.values.astype(np.int64)

    @property
    def opens(self):
        return self.market_opens

    @property
    def closes(self):
        return self.market_closes

    @property
    def break_starts(self):
        return self.market_break_starts

    @property
    def break_ends(self) -> pd.DatetimeIndex:
        return self.market_break_ends

    @property
    def first_session(self) -> pd.Timestamp:
        """First calendar session."""
        return self.sessions[0]

    @property
    def last_session(self) -> pd.Timestamp:
        """Last calendar session."""
        return self.sessions[-1]

    @property
    def first_session_open(self) -> pd.Timestamp:
        """Open time of calendar's first session."""
        return self.opens[0]

    @property
    def last_session_close(self) -> pd.Timestamp:
        """Close time of calendar's last session."""
        return self.closes[-1]

    # TODO: clear duplicated elements
    @staticmethod
    def minute_to_session(dt: pd.Timestamp):
        return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day)

    @staticmethod
    def convert_dt_to_session(dt: pd.Timestamp):
        return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day)

    def find(self, dt: typing.Union[pd.Timestamp, pd.DatetimeIndex]) -> typing.Union[typing.List, int]:
        """
        Find the largest `trade_date` that is less or equal than the `dt`.
        """
        if isinstance(dt, pd.DatetimeIndex):
            ndt = convert_dates_to_int(dt)
        else:
            ndt = convert_date_to_int(dt)
        # Here we should minus one since `arr.searchsorted(dt)` returns the largest i, that satisfies
        # arr[i-1] is less or equal than dt, thus the returned i is the value we wanted add 1.
        # To get the needed index, we should minus one.
        indexes = self.numerical_valid_days.searchsorted(ndt, side="right") - 1
        # Note that if dt is less than all the trade_dates, the returned value is -1, which is not well defined,
        # In this case, we should carefully check this value.
        if indexes.ndim == 0:
            return int(indexes)
        else:
            return list(indexes)

    def shift(self, dt: pd.Timestamp, offset=0) -> pd.Timestamp:
        """
        if offset is 0, then we perform the following operations
        if dt is a trade_date:
            return dt
        else:
            return the largest trade_date that is less than dt.
        In all, shift(dt) returns the largest trade_date that is less or equal than dt.
        $$ sup_{t}\{t: t<=dt, t \in trade_dates \}$$.

        After finding the proper date, we perform the shift operation.
        """
        dt_revision_index = self.find(dt)
        dt_revision_index += offset
        return self.valid_days[dt_revision_index]

    def history(self, count: int, dt: pd.Timestamp) -> pd.DatetimeIndex:
        """
        First we shift the dt to the true
        """
        assert count != 0
        dt_index = self.find(dt)
        return pd.DatetimeIndex(self.valid_days[dt_index-count+1:dt_index+1])

    def sessions_in_range(
            self,
            start_session_label: pd.Timestamp,
            end_session_label: pd.Timestamp,
    ):
        return self.all_sessions[
            self.all_sessions.slice_indexer(
                start_session_label,
                end_session_label
            )
        ]

    # 实现是Hash table, 应该不会花费很多时间
    def session_open(self, session_label: pd.Timestamp):
        return self.schedule.at[
            session_label,
            "market_open"]

    def session_break_start(self, session_label):
        break_start = self.schedule.at[
            session_label,
            'break_start'
        ]
        return break_start

    def session_break_end(self, session_label):
        break_end = self.schedule.at[
            session_label,
            'break_end'
        ]
        return break_end

    def session_close(self, session_label):
        return self.schedule.at[
            session_label,
            'market_close'
        ]

    def is_session(self):
        # TODO: implement this
        raise NotImplementedError


class DateIterator(object):
    def __init__(self, start_date, trading_calendar=None):
        if trading_calendar is None:
            self.trading_calendar = XSHGExchangeCalendar(start_date=start_date)
        self.i = self.trading_calendar.find(start_date)
        if self.i < 0:
            self.i = 0

    @property
    def current_session(self):
        return self.trading_calendar.valid_days[self.i]

    def __next__(self):
        self.i += 1

    def __iter__(self):
        return self


if __name__ == "__main__":
    test = XSHGExchangeCalendar()
    test.find(pd.date_range(start="20050101", periods=10))
    test.history(10, pd.Timestamp("20051110"))
    test.shift(pd.Timestamp("20050104"), 1)
    di = DateIterator(pd.Timestamp("20050101"))
    print(di.current_session)

