import logging
from collections import defaultdict

from torchqtm.data.data_portal import DataPortal
from torchqtm.finance.account import Account
from torchqtm.finance.metrics.metric import Metric
from torchqtm.utils.exchange_calendar import XSHGExchangeCalendar
import numpy as np
import pandas as pd
from torchqtm.types import DATA_FREQUENCIES

log = logging.getLogger(__name__)
import typing


class MetricsTracker:
    """The algorithm's interface to the registered risk and performance metrics

    """

    def __init__(
            self,
            trading_calendar: XSHGExchangeCalendar,
            first_session: pd.Timestamp,
            last_session: pd.Timestamp,
            capital_base: float,
            emission_rate: DATA_FREQUENCIES,
            data_frequency: DATA_FREQUENCIES,
            account: Account,
            metrics: typing.Iterable[Metric],
    ):
        assert isinstance(account, Account), "avoid creating a new account"

        self._benchmark_source = None
        self.emission_rate: typing.Literal["daily", "minute"] = emission_rate
        self._trading_calendar = trading_calendar
        self._first_session: pd.Timestamp = first_session
        self._last_session: pd.Timestamp = last_session
        self._capital_base: float = capital_base
        self._metrics = metrics

        self._current_session = first_session
        self._market_open, self._market_close = self._execution_open_and_close(
            trading_calendar,
            first_session,
        )

        self._session_count = 0
        self._sessions = trading_calendar.sessions_in_range(
            first_session,
            last_session,
        )

        self._total_session_count = len(self._sessions)
        self._account = account

    def _execution_open_and_close(self, trading_calendar, session):
        # TODO: implement this function
        return pd.Timestamp("20050104 09:31:00"), pd.Timestamp("20050104 15:00:00")

    def _progress(self):
        if self.emission_rate == "minute":
            return 1.0

        else:
            return self._session_count / self._total_session_count

    def start_of_simulation(
            self,
            account: Account,
            emission_rate: typing.Literal["daily", "minute"],
            trading_calendar,
            sessions: pd.DatetimeIndex,
            benchmark_source,
    ):
        for metric in self._metrics:
            metric.start_of_simulation(
                account=account,
                emission_rate=emission_rate,
                trading_calendar=trading_calendar,
                sessions=sessions,
                benchmark_source=benchmark_source,
            )

    def end_of_simulation(
            self,
            packet: typing.Dict,
            account: Account,
            trading_calendar,
            sessions: pd.DatetimeIndex,
            data_portal: DataPortal,
            benchmark_source,
    ):
        for metric in self._metrics:
            metric.end_of_simulation(
                packet=packet,
                account=account,
                trading_calendar=trading_calendar,
                sessions=sessions,
                data_portal=data_portal,
                benchmark_source=benchmark_source,
            )

    def start_of_session(
            self,
            account: Account,
            session: pd.Timestamp,
            data_portal: DataPortal,
    ):
        for metric in self._metrics:
            metric.start_of_session(
                account=account,
                session=session,
                data_portal=data_portal,
            )

    def end_of_session(
            self,
            packet: typing.Dict,
            account: Account,
            session: pd.Timestamp,
            session_idx: int,
            data_portal: DataPortal,
    ):
        for metric in self._metrics:
            metric.end_of_session(
                packet=packet,
                account=account,
                session=session,
                session_idx=session_idx,
                data_portal=data_portal,
            )

    def start_of_bar(self, *args, **kwargs):
        for metric in self._metrics:
            metric.start_of_bar(*args, **kwargs)

    def end_of_bar(
            self,
            packet: typing.Dict,
            account: Account,
            dt: pd.Timestamp,
            session_idx: int,
            data_portal: DataPortal,
    ):
        for metric in self._metrics:
            metric.end_of_bar(
                packet=packet,
                account=account,
                dt=dt,
                session_idx=session_idx,
                data_portal=data_portal,
            )

    def handle_start_of_simulation(self, benchmark_source):
        self._benchmark_source = benchmark_source
        self.start_of_simulation(
            self._account,
            self.emission_rate,
            self._trading_calendar,
            self._sessions,
            benchmark_source,
        )

    def handle_start_of_session(self, session_label: pd.Timestamp, data_portal: DataPortal):
        self._account.start_of_session(session_label)

        # TODO: load and process dividends
        # adjustment_reader = data_portal.adjustment_reader
        # if adjustment_reader is not None:
        #     # this is None when running with a dataframe source
        #     ledger.process_dividends(
        #         session_label,
        #         self._asset_finder,
        #         adjustment_reader,
        #     )
        self._current_session = session_label
        self._market_open, self._market_close = self._execution_open_and_close(
            self._trading_calendar,
            session_label,
        )

        self.start_of_session(
            account=self._account,
            session=session_label,
            data_portal=data_portal,
        )

        # handle any splits that impact any positions or any open orders.
        assets_we_care_about = (
                self._account.positions.keys() | self._account.orders_tracker.open_orders.keys()
        )

        if assets_we_care_about:
            splits = data_portal.get_splits(assets_we_care_about, session_label)
            if splits:
                # 订单要进行分割, metrics_tracker也能进行分割???? 其中谁发挥了作用
                self._account.orders_tracker.handle_splits(splits)
                self._account.handle_splits(splits)

    def handle_end_of_session(self, dt, data_portal):
        """Handles the close of the given day.

        Parameters
        ----------
        dt : Timestamp
            The most recently completed simulation datetime.
        data_portal : DataPortal
            The current data portal.

        Returns
        -------
        A daily perf packet.
        """
        completed_session = self._current_session

        if self.emission_rate == "daily":
            # this method is called for both minutely and daily emissions, but
            # this chunk of code here only applies for daily emissions. (since
            # it's done every minute, elsewhere, for minutely emission).
            self._account.sync_last_sale_price(dt, data_portal)

        session_ix = self._session_count
        # increment the day counter before we move markers forward.
        self._session_count += 1

        packet = {
            "period_start": self._first_session,
            "period_end": self._last_session,
            "capital_base": self._capital_base,
            "daily_perf": {
                "period_open": self._market_open,
                "period_close": dt,
            },
            "cumulative_perf": {
                "period_open": self._first_session,
                "period_close": self._last_session,
            },
            "progress": self._progress(),
            "cumulative_risk_metrics": {},
        }
        self._account.end_of_session(session_ix)
        self.end_of_session(
            packet,
            self._account,
            completed_session,
            session_ix,
            data_portal,
        )
        return packet

    def handle_simulation_end(self, data_portal: DataPortal):
        """When the simulation is complete, run the full period risk report
        and send it out on the results socket.
        """
        log.info(
            "Simulated %(days)s trading days\n first open: %(first)s\n last close: %(last)s",
            dict(
                days=self._session_count,
                first=self._trading_calendar.session_open(self._first_session),
                last=self._trading_calendar.session_close(self._last_session),
            ),
        )

        packet = {}
        self.end_of_simulation(
            packet,
            self._account,
            self._trading_calendar,
            self._sessions,
            data_portal,
            self._benchmark_source,
        )
        return packet