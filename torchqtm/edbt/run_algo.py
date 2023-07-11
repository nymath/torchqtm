import pandas as pd
import typing

from torchqtm.edbt.algorithm import TradingAlgorithm
from torchqtm.edbt.sim_params import SimulationParameters

from torchqtm.finance.account import Account
from torchqtm.utils.calendar_utils import get_calendar
from torchqtm.utils.datetime_utils import DateTimeManager
from torchqtm.types import DATA_FREQUENCIES
from torchqtm.data.data_portal import DataPortal

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

        benchmark_symbol, benchmark_returns = benchmark_spec.resolve(start, end)

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
            **account_configs,
        )

        algo = Algo(
            sim_params=sim_params,
            data_portal=data_portal,
            namespace=None,
            trading_calendar=datetime_manager.trading_calendar,
            datetime_manager=datetime_manager,
            metrics_set=None,
            benchmark_returns=benchmark_returns,
            account=account,
        )

