import logging

import pandas as pd

log = logging.getLogger("SimulationParameters")

DEFAULT_CAPITAL_BASE = 1e5
from torchqtm.utils.exchange_calendar import XSHGExchangeCalendar
from torchqtm.types import DATA_FREQUENCIES
from functools import lru_cache


class SimulationParameters(object):
    """
    The collections of parameters for creating backtesting environments.
    """
    def __init__(
            self,
            start_session: pd.Timestamp,
            end_session: pd.Timestamp,
            trading_calendar: XSHGExchangeCalendar,
            capital_base: float = DEFAULT_CAPITAL_BASE,
            emission_rate: DATA_FREQUENCIES = "daily",
            data_frequency: DATA_FREQUENCIES = "daily",
    ):
        assert type(start_session) == pd.Timestamp
        assert type(end_session) == pd.Timestamp

        assert trading_calendar is not None, "Must pass in trading calendar!"
        assert start_session <= end_session, "Period start falls after period end."

        assert start_session >= trading_calendar.first_session
        assert end_session <= trading_calendar.last_session

        self._start_session = start_session
        self._end_session = end_session

        self._capital_base = capital_base

        self._emission_rate = emission_rate
        self._data_frequency = data_frequency

        self._trading_calendar = trading_calendar

        # if the start_date is not the trading_session(Almost never happen)

        self._start_session = trading_calendar.minute_to_session(self._start_session)

        self._end_session = trading_calendar.minute_to_session(self._end_session)

    @property
    def capital_base(self):
        return self._capital_base

    @property
    def emission_rate(self):
        return self._emission_rate

    @property
    def data_frequency(self):
        return self._data_frequency

    def set_data_frequency(self, val):
        self._data_frequency = val

    @property
    def start_session(self):
        return self._start_session

    @property
    def end_session(self):
        return self._end_session

    @property
    def trading_calendar(self):
        return self._trading_calendar

    def create_new(self, start_session, end_session, data_frequency=None):
        raise NotImplementedError

    @property
    @lru_cache(None)
    def sessions(self):
        return self._trading_calendar.sessions_in_range(self._start_session, self._end_session)

    def __repr__(self):
        template = """
{class_name}(
    start_session={start_session},
    end_session={end_session},
    capital_base={capital_base},
    data_frequency={data_frequency},
    emission_rate={emission_rate},
    trading_calendar={trading_calendar},
)
""".strip()
        return template.format(
            class_name=self.__class__.__name__,
            start_session=self.start_session,
            end_session=self.end_session,
            capital_base=self.capital_base,
            data_frequency=self.data_frequency,
            emission_rate=self.emission_rate,
            trading_calendar=self.trading_calendar,
        )


if __name__ == "__main__":
    start_date = pd.Timestamp("20100101")
    end_date = pd.Timestamp("20100201")
    from torchqtm.utils.calendar_utils import get_calendar
    test = SimulationParameters(start_date, end_date, get_calendar("XSHG"))
    test.sessions
    repr(test)
    print(test)
    print(test.start_session)
    print(test.end_session)




