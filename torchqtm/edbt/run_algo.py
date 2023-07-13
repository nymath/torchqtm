import pandas as pd
import typing

from torchqtm.edbt.algorithm import TradingAlgorithm
from torchqtm.edbt.sim_params import SimulationParameters

from torchqtm.finance.account import Account
from torchqtm.finance.metrics.tracker import MetricsTracker
from torchqtm.utils.calendar_utils import get_calendar
from torchqtm.utils.datetime_utils import DateTimeManager
from torchqtm.types import DATA_FREQUENCIES
from torchqtm.data.data_portal import DataPortal
from torchqtm.assets import Equity
from torchqtm.finance.metrics.loader import DEFAULT_METRICS
# Design a time machine


class run_algo(object):
    def __init__(
            self,
            Algo: typing.Type[TradingAlgorithm],
            data_frequency: DATA_FREQUENCIES,
            capital_base: float,
            bundle: str,
            start: pd.Timestamp,
            end: pd.Timestamp,
            output,
            trading_calendar,
            metrics_set,
            local_namespace,
            environ,
            account_configs,
            benchmark_spec,
    ):

        if trading_calendar is None:
            trading_calendar = get_calendar("XSHG")
        datetime_manager = DateTimeManager(trading_calendar)

        # benchmark_symbol, benchmark_returns = benchmark_spec.resolve(start, end)
        benchmark_symbol, benchmark_returns = (0, 0)

        data_portal = DataPortal(
            data_frequency=data_frequency,
            trading_calendar=datetime_manager.trading_calendar,
            datetime_manager=datetime_manager,
            restrictions=None,
        )

        sim_params = SimulationParameters(
            start_session=start,
            end_session=end,
            trading_calendar=trading_calendar,
            capital_base=capital_base,
            data_frequency=data_frequency,
        )

        account = Account(
            datetime_manager=datetime_manager,
            trading_sessions=trading_calendar.sessions_in_range(start, end),
            capital_base=capital_base,
            data_frequency=data_frequency,
        )

        metrics_tracker = MetricsTracker(
            trading_calendar=trading_calendar,
            first_session=start,
            last_session=end,
            capital_base=capital_base,
            emission_rate=data_frequency,
            data_frequency=data_frequency,
            account=account,
            metrics=metrics_set,
        )

        algo = Algo(
            sim_params=sim_params,
            data_portal=data_portal,
            namespace=None,
            trading_calendar=datetime_manager.trading_calendar,
            datetime_manager=datetime_manager,
            benchmark_returns=benchmark_returns,
            account=account,
            metrics_tracker=metrics_tracker,
        )

        it = iter(algo.get_generator())
        while True:
            try:
                next(it)
            except StopIteration:
                break

        print("Hello World")


class TestAlgo(TradingAlgorithm):
    def initialize(self):
        self.safe_set_attr("sym", Equity("000001.XSHE"))
        self.safe_set_attr("count", 0)

    def before_trading_start(self):
        pass

    def handle_data(self):
        if self.count == 0:
            self.order(self.sym, 10000)

        print("----------------------------------------------------")
        print("current_dt", self.current_dt, self.data_portal.get_spot_value([self.sym], "close", self.current_dt, "daily"))
        # print("orders", self.account.orders_tracker.data)
        print("cash", self.account.ledger.cash)
        self.data_portal.history_bars(self.sym, ["close"], 200, "daily")
        print("portfolio_value", self.account.ledger.portfolio_value)
        print("positions_value", self.account.ledger.positions_value)
        print("returns", self.account.ledger.returns)
        print(self.account.ledger.cash_flow)
        self.count += 1

    def analyze(self):
        self.current_dt


if __name__ == "__main__":
    Algo = TestAlgo
    data_frequency: DATA_FREQUENCIES = "daily"
    capital_base = 1e6
    bundle = "rqalpha"
    start = pd.Timestamp("20220101")
    end = pd.Timestamp("20230101")
    output = None
    trading_calendar = None
    metrics_set = DEFAULT_METRICS
    local_namespace = None
    environ = None
    account_configs = None
    benchmark_spec = None

    run_algo(
        Algo=Algo,
        data_frequency=data_frequency,
        capital_base=capital_base,
        bundle=bundle,
        start=start,
        end=end,
        output=output,
        trading_calendar=trading_calendar,
        metrics_set=metrics_set,
        local_namespace=local_namespace,
        environ=environ,
        account_configs=account_configs,
        benchmark_spec=benchmark_spec,
    )
