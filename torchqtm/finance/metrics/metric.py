import datetime
import typing
from functools import partial
from abc import ABCMeta, abstractmethod

from torchqtm.data.data_portal import DataPortal
from torchqtm.finance.account import Account
import empyrical as ep
import numpy as np
import pandas as pd

# 这里我不是很想去写这个, 因为目前参数命名不规范, 我后边自己再改改
# 另外我感觉使用handle_event可能比他更好


class StatsPacket(object):
    def __init__(self):
        self.cumulative_risk_metrics = {}


class Metric(object, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def start_of_simulation(
            cls,
            account: Account,
            emission_rate: typing.Literal["daily", "minute"],
            trading_calendar,
            sessions: pd.DatetimeIndex,
            benchmark_source,
    ):
        pass

    @classmethod
    def end_of_simulation(
            cls,
            packet: typing.Dict,
            account: Account,
            trading_calendar,
            sessions: pd.DatetimeIndex,
            data_portal: DataPortal,
            benchmark_source,
    ):
        pass

    @classmethod
    def start_of_session(
            cls,
            account: Account,
            session: pd.Timestamp,
            data_portal: DataPortal,
    ):
        pass

    @classmethod
    def end_of_session(
            cls,
            packet: typing.Dict,
            account: Account,
            session: pd.Timestamp,
            session_idx: int,
            data_portal: DataPortal,
    ):
        pass

    @classmethod
    def start_of_bar(
            cls,
            packet: typing.Dict,
            account: Account,
            dt: pd.Timestamp,
            session_idx: int,
            data_portal: DataPortal,
    ):
        pass

    @classmethod
    def end_of_bar(
            cls,
            packet: typing.Dict,
            account: Account,
            dt: pd.Timestamp,
            session_idx: int,
            data_portal: DataPortal,
    ):
        pass


class Returns(Metric):
    @classmethod
    def end_of_bar(cls, packet, account, dt, session_idx, data_portal):
        packet["minute_perf"]["returns"] = account.todays_returns
        packet["cumulative_perf"]["returns"] = account.ledger.returns
        packet["cumulative_risk_metrics"]["algorithm_period_return"] = account.ledger.returns

    @classmethod
    def end_of_session(cls, packet, account, session, session_idx, data_portal):
        packet["daily_perf"]["returns"] = account.todays_returns
        packet["cumulative_perf"]["returns"] = account.ledger.returns
        packet["cumulative_risk_metrics"]["algorithm_period_return"] = account.ledger.returns


class PNL(Metric):
    """Tracks daily and cumulative PNL."""
    _previous_pnl: float = 0.0

    @classmethod
    def start_of_simulation(cls, account, emission_rate, trading_calendar, sessions, benchmark_source):
        cls._previous_pnl = 0.0

    @classmethod
    def start_of_session(cls, account: Account, session, data_portal):
        cls._previous_pnl = account.ledger.pnl

    @classmethod
    def _end_of_period(cls, field, packet, account):
        pnl = account.ledger.pnl
        packet[field]["pnl"] = pnl - cls._previous_pnl
        packet["cumulative_perf"]["pnl"] = account.ledger.pnl

    @classmethod
    def end_of_bar(cls, packet, account, dt, session_idx, data_portal):
        pnl = account.ledger.pnl
        packet["minute_perf"]["pnl"] = pnl - cls._previous_pnl
        packet["cumulative_perf"]["pnl"] = account.ledger.pnl

    @classmethod
    def end_of_session(cls, packet, account, session, session_idx, data_portal):
        pnl = account.ledger.pnl
        packet["daily_perf"]["pnl"] = pnl - cls._previous_pnl
        packet["cumulative_perf"]["pnl"] = account.ledger.pnl


class Orders(Metric):

    def end_of_bar(self, packet, account, dt, session_idx, data_portal):
        packet["minute_perf"]["orders"] = account.get_orders(dt)

    def end_of_session(self, packet, account, dt, session_idx, data_portal):
        packet["daily_perf"]["orders"] = account.get_orders()


class Transactions(Metric):
    """Tracks daily transactions."""

    def end_of_bar(self, packet, account, dt, session_idx, data_portal):
        packet["minute_perf"]["transactions"] = account.get_transactions(dt)

    def end_of_session(self, packet, account, dt, session_idx, data_portal):
        packet["daily_perf"]["transactions"] = account.get_transactions()


# class Positions(Metric):
#
#     def end_of_bar(
#             self,
#             packet: typing.Dict,
#             account: Account,
#             dt: pd.Timestamp,
#             session_idx: int,
#             data_portal: DataPortal,
#     ):
#         packet["minute_perf"]["positions"] = account.get_positions(dt)
#
#     def end_of_session(
#             self,
#             packet: typing.Dict,
#             account: Account,
#             dt: pd.Timestamp,
#             session_idx: int,
#             data_portal: DataPortal,
#     ):
#         packet["daily_perf"]["positions"] = account.get_positions()


# 这个类看起来不太好, 这个操纵明显可以通过向量化完成
class ReturnsStatistics(Metric):
    """A metric that reports an end of simulation scalar or time series
    computed from the algorithm returns

    Parameters
    ----------
    function: callable
        A function that accepts a numpy array of returns and returns a scalar
        or time series
    filed_name: str, optional
        The name of the field. If not provided, it will be
        ``function.__name__``.
    """
    def __init__(self, function: typing.Callable, filed_name: str):
        super().__init__()
        if filed_name is None:
            filed_name = function.__name__

        self._function = function
        self._filed_name = filed_name

    def end_of_bar(self, packet, account, dt, session_idx, data_portal):
        res = self._function(account.daily_returns_series.values[: session_idx + 1])
        if not np.isfinite(res):
            res = None
        packet["cumulative_risk_metrics"][self._filed_name] = res

    def end_of_session(self, packet, account, dt, session_idx, data_portal):
        res = self._function(account.daily_returns_series.values[: session_idx + 1])
        if not np.isfinite(res):
            res = None
        packet["cumulative_risk_metrics"][self._filed_name] = res


class AlphaBeta(Metric):
    def start_of_simulation(
            self,
            account: Account,
            emission_rate: typing.Literal["daily", "minute"],
            trading_calendar,
            sessions: pd.DatetimeIndex,
            benchmark_source,
    ):
        pass


class NumTradingDays(Metric):

    _num_trading_days: int = 0

    def start_of_simulation(
            self,
            account: Account,
            emission_rate: typing.Literal["daily", "minute"],
            trading_calendar,
            sessions: pd.DatetimeIndex,
            benchmark_source,
    ):
        self._num_trading_days = 0

    def start_of_session(
            self,
            account: Account,
            session: pd.Timestamp,
            data_portal: DataPortal,
    ):
        self._num_trading_days += 1

    def end_of_bar(
            self,
            packet: typing.Dict,
            account: Account,
            dt: pd.Timestamp,
            session_idx: int,
            data_portal: DataPortal,
    ):
        packet["cumulative_risk_metrics"]["trading_days"] = self._num_trading_days





