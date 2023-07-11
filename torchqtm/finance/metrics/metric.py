import datetime
from functools import partial
from abc import ABCMeta, abstractmethod

import empyrical as ep
import numpy as np
import pandas as pd

# 这里我不是很想去写这个, 因为目前参数命名不规范, 我后边自己再改改
# 另外我感觉使用handle_event可能比他更好


class Metric(object, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    def start_of_simulation(self, account, emission_rate, trading_calendar, sessions, benchmark_source):
        pass

    def end_of_simulation(self, packet, account, trading_calendar, sessions, data_portal, benchmark_source):
        pass

    def start_of_session(self, account, session, data_portal):
        pass

    def end_of_session(self, packet, account, session, session_ix, data_portal):
        pass

    def start_of_bar(self, packet, account, dt, session_ix, data_portal):
        pass

    def end_of_bar(self, packet, account, dt, session_ix, data_portal):
        pass


class Returns(Metric):
    def _end_of_period(field, packet, account, dt, session_ix, data_portal):
        packet[field]["returns"] = account.todays_returns
        packet["cumulative_perf"]["returns"] = account.returns
        packet["cumulative_risk_metrics"][
            "algorithm_period_return"
        ] = account.returns

    end_of_bar = partial(_end_of_period, "minute_perf")
    end_of_session = partial(_end_of_period, "daily_perf")


class PNL:
    """Tracks daily and cumulative PNL."""

    def start_of_simulation(
        self, ledger, emission_rate, trading_calendar, sessions, benchmark_source
    ):
        self._previous_pnl = 0.0

    def start_of_session(self, account, session, data_portal):
        self._previous_pnl = account.portfolio.pnl

    def _end_of_period(self, field, packet, account):
        pnl = account.pnl
        packet[field]["pnl"] = pnl - self._previous_pnl
        packet["cumulative_perf"]["pnl"] = account.pnl

    def end_of_bar(self, packet, account, dt, session_ix, data_portal):
        self._end_of_period("minute_perf", packet, account)

    def end_of_session(self, packet, account, session, session_ix, data_portal):
        self._end_of_period("daily_perf", packet, account)


class Transactions:
    """Tracks daily transactions."""

    def end_of_bar(self, packet, account, dt, session_ix, data_portal):
        packet["minute_perf"]["transactions"] = account.get_transactions(dt)

    def end_of_session(self, packet, account, dt, session_ix, data_portal):
        packet["daily_perf"]["transactions"] = account.get_transactions()

