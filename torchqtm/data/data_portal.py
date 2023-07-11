from functools import lru_cache

from torchqtm.data.rq_data_source.adjust import convert_date_to_int
from torchqtm.utils.exchange_calendar import XSHGExchangeCalendar, DateIterator
from torchqtm.assets import Asset
import numpy as np
import pandas as pd
import typing

from torchqtm.types import (
    BASE_FIELDS,
    HISTORY_FREQUENCIES,
    ASSET_TYPE,
)

from torchqtm.data.rq_data_source.data_source import (
    ReadersTracker,
)

from torchqtm.data.rq_data_source.readers import StockDailyBarReader
from torchqtm.utils.datetime_utils import DateTimeManager, DateTimeMixin


def _find_position_of_dt(bars: np.ndarray, dt: pd.Timestamp):
    dt = np.uint64(convert_date_to_int(dt))
    pos = bars['datetime'].searchsorted(dt)
    if pos >= len(bars) or bars['datetime'][pos] != dt:
        return None
    return pos


def convert_dt_to_session_label(dt: pd.Timestamp):
    return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day)


class DataPortal(object, DateTimeMixin):
    # TODO: add cache for quick access to history data
    def __init__(self,
                 data_source=ReadersTracker(),
                 data_frequency: typing.Literal["daily", "minute"] = "daily",
                 trading_calendar: XSHGExchangeCalendar = None,
                 datetime_manager: DateTimeManager = None,
                 restrictions=None):
        DateTimeMixin.__init__(self, datetime_manager)

        self._data_source = data_source
        self._datetime = None
        self.data_frequency = data_frequency
        self._views = {}
        self._daily_mode = (self.data_frequency == "daily")

        self._adjust_minutes = False
        if trading_calendar is None:
            self.trading_calendar = XSHGExchangeCalendar()
        else:
            self.trading_calendar = trading_calendar
        self._date_iterator = DateIterator(start_date=self.trading_calendar.start_date)
        # self._is_restricted = restrictions.is_restricted

    @property
    def current_session(self):
        return self._date_iterator.current_session

    def current(self, asset, filed):
        return self.get_scalar_asset_spot_value(asset, filed, self.current_dt, self.data_frequency)

    def _ensure_reader_aligned(self, reader):
        pass

    def _reindex_extra_aligned(self, reader):
        pass

    def _reindex_extra_source(self, dt, source_date_index):
        pass

    def handle_extra_source(self, source_df, sim_params):
        pass

    def _get_pricing_reader(self, data_frequency):
        pass

    def get_last_traded_dt(
            self,
            asset,
            dt,
            data_frequency,
    ):
        pass

    def _get_daily_spot_value(self, asset, field, session_label: pd.Timestamp) -> typing.Optional[float]:
        bars = self._data_source.get_bar(asset, session_label, "1d")
        if bars is None:
            return None
        else:
            return float(bars[field])

    def get_spot_value(
            self,
            assets: typing.Iterable[Asset],
            field: str,
            dt: pd.Timestamp,
            data_frequency: str,
    ):
        """
        >>>self.get_spot_value([asset], 'close', dt, 'daily')
        106.25
        """
        if data_frequency != '1d':
            raise NotImplementedError

        if data_frequency == "daily":
            session_label = self.trading_calendar.convert_dt_to_session(dt)
            return [self._get_daily_spot_value(asset, field, session_label) for asset in assets]

    def get_scalar_asset_spot_value(
        self,
        asset: Asset,
        filed: str,
        dt: pd.Timestamp,
        data_frequency: typing.Literal["daily", "minute"],
    ):
        # """
        # >>>self.get_spot_value([asset], 'close', dt, 'daily')
        # 106.25
        # """
        if data_frequency != "daily":
            return NotImplementedError

        return self._get_daily_spot_value(asset, filed, self.trading_calendar.convert_dt_to_session(dt))

    def get_adjustments(self, assets: typing.List[Asset], dt: pd.Timestamp, perspective_dt):
        """
        perspective_dt : pd.Timestamp
            The timestamp from which the data is being viewed back from.
        >>>self.get_adjustments([asset], "close", dt, pd.Timestamp('20160102'))
        [0.9832470935131806]
        """
        pass

    def get_adjusted_value(
            self,
            asset: Asset,
            dt: pd.Timestamp,
            data_frequency: HISTORY_FREQUENCIES,
            spot_value: typing.Optional[float] = None,
    ):
        """
        >>>self.get_adjusted_value(asset, "close", dt, pd.Timestamp('20160102'), "daily")
        104.47000368577544
        """

    def _get_minute_spot_value(self, asset, column, dt, ffill=False):
        pass

    def _get_days_for_window(self, end_date, bar_count):
        """
        >>>self._get_days_for_window(dt, 5)
        DatetimeIndex(['2014-12-29', '2014-12-30', '2014-12-31', '2015-01-02',
                       '2015-01-05'],
                      dtype='datetime64[ns]', freq='C')
        """

    def _get_history_daily_window(
            self,
            assets,
            end_dt,
            bar_count,
            field,
            data_frequency,
    ):
        """
        >>>self._get_history_daily_window([asset], dt, 5, "close", "daily")
        Out[53]:
                    Equity(8 [AAPL])
        2014-12-29            113.91
        2014-12-30            112.52
        2014-12-31            110.38
        2015-01-02            109.33
        2015-01-05            106.25
        """
        bars = self._all_daily_bars_of(asset)
        bars[field]

    def _get_history_daily_window_data(
            self,
            assets,
            days_for_window,
            end_dt,

    ):
        """
        这个应该用不太上了
        """
        pass

    def handle_minute_history_out_of_bounds(self):
        pass

    def _get_history_minute_window(self):
        pass

    def get_history_window(
            self,
            assets: typing.List[Asset],
            end_dt: pd.Timestamp,
            bar_count: int,
            frequency: typing.Literal["1d", "1m"],
            field: BASE_FIELDS,
            ffill: bool = False,
    ) -> pd.DataFrame:
        """
        不太明白
        """
        pass

    def _get_minute_window_data(self, assets, field, minutes_for_window):
        pass

    def _get_daily_window_data(self, assets, field, days_in_window, extra_slot=True):
        pass

    def _get_adjustment_list(self, asset, adjustments_dict, table_name):
        pass

    def get_splits(self, assets, dt):
        """
        """
        pass

    def get_stock_dividends(self, sid, trading_days):
        pass

