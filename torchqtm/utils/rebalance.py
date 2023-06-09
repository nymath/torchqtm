from abc import ABCMeta, abstractmethod
from pandas_market_calendars import get_calendar
import itertools
from collections import defaultdict
from typing import Sized, Iterable


class Calendar(object, metaclass=ABCMeta):
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.exchange = 'SSE'
        self.calendar = get_calendar(self.exchange)
        self.trade_dates = self.calendar.valid_days(start_date=self.start_date, end_date=self.end_date)
        self.trade_dates = self.trade_dates.tz_convert(None)

    def create_monthly_groups(self):
        rlt = defaultdict(list)
        for key, value in zip(self.trade_dates, [(x.year, x.month) for x in self.trade_dates]):
            rlt[value].append(key)
        return rlt

    def create_weekly_groups(self):
        rlt = defaultdict(list)
        for key, value in zip(self.trade_dates, [(x.year, x.week) for x in self.trade_dates]):
            rlt[value].append(key)
        return rlt


class Daily(object):
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_dates = None
        self._create_trade_dates()

    def _create_trade_dates(self):
        calendar = Calendar(self.start_date, self.end_date)
        self.rebalance_dates = calendar.trade_dates

    @property
    def data(self):
        return self.rebalance_dates


class Weekly(object):
    def __init__(self, start_date, end_date, days: Iterable[int]):
        self.start_date = start_date
        self.end_date = end_date
        self.days = days
        self.rebalance_dates = None
        self._create_trade_dates()

    def _create_trade_dates(self):
        calendar = Calendar(self.start_date, self.end_date)
        temp = calendar.create_weekly_groups()
        self.rebalance_dates = sorted([x[i] for x in temp.values() for i in self.days])

    @property
    def data(self):
        return self.rebalance_dates


class Monthly(object):
    def __init__(self, start_date, end_date, days: Iterable[int]):
        self.start_date = start_date
        self.end_date = end_date
        self.days = days
        self.rebalance_dates = None
        self._create_trade_dates()

    def _create_trade_dates(self):
        calendar = Calendar(self.start_date, self.end_date)
        temp = calendar.create_weekly_groups()
        self.rebalance_dates = sorted([x[i] for x in temp.values() for i in self.days])

    @property
    def data(self):
        return self.rebalance_dates
