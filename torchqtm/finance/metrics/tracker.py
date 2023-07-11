import logging
from collections import defaultdict

from torchqtm.data.data_portal import DataPortal
from torchqtm.finance.account import Account
from torchqtm.utils.exchange_calendar import XSHGExchangeCalendar
import numpy as np
import pandas as pd
from torchqtm.types import DATA_FREQUENCIES

log = logging.getLogger(__name__)
import typing


class MetricsTracker:
    """The algorithm's interface to the registered risk and performance metrics

    """
    _hooks = (
        "start_of_simulation",
        "end_of_simulation",
        "start_of_session",
        "end_of_session",
        "start_of_bar",
        "end_of_bar",
    )

    def __init__(
            self,
            trading_calendar: XSHGExchangeCalendar,
            first_session,
            last_session,
            capital_base,
            emission_rate: DATA_FREQUENCIES,
            data_frequency: DATA_FREQUENCIES,
            metrics,
    ):
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
        self._account = Account(self._sessions, capital_base, data_frequency)

        if emission_rate == "minute":
            def progress(self):
                return 1.0  # a fake value
        else:

            def progress(self):
                return self._session_count / self._total_session_count

        self._progress = progress

    def start_of_simulation(self, *args, **kwargs):
        for metric in self._metrics:
            metric.start_of_simulation(*args, **kwargs)

    def end_of_simulation(self, *args, **kwargs):
        for metric in self._metrics:
            metric.end_of_simulation(*args, **kwargs)

    def start_of_session(self, *args, **kwargs):
        for metric in self._metrics:
            metric.start_of_session(*args, **kwargs)

    def end_of_session(self, *args, **kwargs):
        for metric in self._metrics:
            metric.end_of_session(*args, **kwargs)

    def start_of_bar(self, *args, **kwargs):
        for metric in self._metrics:
            metric.start_of_bar(*args, **kwargs)

    def end_of_bar(self, *args, **kwargs):
        for metric in self._metrics:
            metric.end_of_bar(*args, **kwargs)

    def handle_start_of_simulation(self, benchmark_source):
        self._benchmark_source = benchmark_source
        self.start_of_simulation(
            self._account,
            self.emission_rate,
            self._trading_calendar,
            self._sessions,
            benchmark_source,
        )

    def handle_market_close(self, dt, data_portal):
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
            self.sync_last_sale_prices(dt, data_portal)

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
            "progress": self._progress(self),
            "cumulative_risk_metrics": {},
        }
        ledger = self._ledger
        ledger.end_of_session(session_ix)
        self.end_of_session(
            packet,
            ledger,
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
            self._ledger,
            self._trading_calendar,
            self._sessions,
            data_portal,
            self._benchmark_source,
        )
        return packet



