from abc import ABCMeta, abstractmethod
from pandas_market_calendars import get_calendar
import itertools
from collections import defaultdict


# 改进
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


if __name__ == '__main__':
    calendar = Calendar('20220101', '20220201')
    print(calendar.create_monthly_groups())

