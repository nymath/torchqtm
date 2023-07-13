import pandas as pd
import numpy as np

import datetime

from torchqtm.constants import (
    GLOBAL_DEFAULT_MARKET_OPEN,
    GLOBAL_DEFAULT_MARKET_CLOSE,
    GLOBAL_DEFAULT_BEFORE_TRADING_START_OFFSET,
)

from torchqtm.types import EVENT_TYPE


class DailySimulationClock(object):
    def __init__(
            self,
            sessions: pd.DatetimeIndex,
            market_open: datetime.time = GLOBAL_DEFAULT_MARKET_OPEN,
            market_close: datetime.time = GLOBAL_DEFAULT_MARKET_CLOSE,
            offset: int = GLOBAL_DEFAULT_BEFORE_TRADING_START_OFFSET,
    ):
        self.sessions_nanos = sessions.values.astype(np.int64)
        duration_0 = (market_open.hour * 60 + market_open.minute - offset) * 60 * 1e9
        duration_1 = (market_close.hour * 60 + market_close.minute) * 60 * 1e9
        self.bts_nanos = self.sessions_nanos + duration_0
        self.market_close_nanos = self.sessions_nanos + duration_1

    def __iter__(self):
        for idx in range(len(self.sessions_nanos)):
            session_nano = self.sessions_nanos[idx]
            bts_nano = self.bts_nanos[idx]
            market_close_nano = self.market_close_nanos[idx]
            yield pd.Timestamp(session_nano), EVENT_TYPE.SESSION_START
            yield pd.Timestamp(bts_nano), EVENT_TYPE.BEFORE_TRADING_START
            yield pd.Timestamp(market_close_nano), EVENT_TYPE.BAR
            yield pd.Timestamp(market_close_nano), EVENT_TYPE.SESSION_END


class MinuteSimulationClock(object):
    def __init__(self,
                 sessions: pd.DatetimeIndex,
                 market_open: datetime.time,
                 market_close: datetime.time,
                 offset: int):
        self.sessions_nanos = sessions.values.astype(np.int64)
        duration_0 = (market_open.hour * 60 + market_open.minute - offset) * 60 * 1e9
        duration_1 = (market_close.hour * 60 + market_close.minute) * 60 * 1e9
        self.bts_nanos = self.sessions_nanos + duration_0
        self.market_close_nanos = self.sessions_nanos + duration_1

    def __iter__(self):
        # for idx in range(len(self.sessions_nanos)):
        #     session_nano = self.sessions_nanos[idx]
        #     bts_nano = self.bts_nanos[idx]
        #     market_close_nano = self.market_close_nanos[idx]
        #     yield pd.Timestamp(session_nano), Event.SESSION_START
        #     yield pd.Timestamp(bts_nano), Event.BEFORE_TRADING_START
        #     yield pd.Timestamp(market_close_nano), Event.BAR
        #     yield pd.Timestamp(market_close_nano), Event.SESSION_END
        pass
