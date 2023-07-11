import pandas as pd

from torchqtm.utils.calendar_utils import get_calendar
from torchqtm.types import DATA_FREQUENCIES
from torchqtm.data.data_portal import DataPortal


def _run(
    data_frequency: DATA_FREQUENCIES,
    bundle: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output,
    trading_calendar,
    metrics_set,
    local_namespace,
    environ,
    account,
    benchmark_spec,
):

    if trading_calendar is None:
        trading_calendar = get_calendar("XSHG")

    benchmark_symbol, benchmark_returns = benchmark_spec.resolve(start, end)

    data_portal = DataPortal(

    )




