import numpy as np
import pandas as pd
from functools import lru_cache

class DataProxy(object):
    def __init__(self, data_source, price_board):
        self._data_source = data_source
        self._price_board = price_board
        trading_calendars = data_source.get_trading_calendars()

    def __getattr__(self, item):
        return getattr(self._data_source, item)

    def get_trading_minutes_for(self, order_book_id, dt):
        pass

    def get_yield_curve(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Out[29]:
                          0S        1M        2M  ...       30Y       40Y       50Y
        2010-01-04  0.008985  0.010646  0.012036  ...  0.041856  0.042444  0.043044
        2010-01-05  0.008960  0.010462  0.011965  ...  0.041880  0.042468  0.043068
        """
        pass

    def get_risk_free_rate(self, start_date, end_date):
        pass

    def get_dividend(self, order_book_id):
        pass

    def get_split(self, order_book_id):
        pass

    def get_dividend_by_book_date(self, order_book_id, date):
        pass

    def get_split_by_ex_date(self, order_book_id, date):
        pass

    @lru_cache(10240)
    def _get_prev_close(self, order_book_id, dt):
        """
        >>>dp._get_prev_close("000001.XSHG", pd.Timestamp('20160101'))
        Out[41]: 3539.1819
        """
        pass

    def get_prev_close(self, order_book_id, dt):
        return self._get_prev_close(order_book_id, dt.replace(hour=0, minute=0, second=0))

    @lru_cache(10240)
    def _get_prev_settlement(self, instrument, dt):
        pass

    @lru_cache(10240)
    def _get_settlement(self, instrument, dt):
        pass

    def get_prev_settlement(self, order_book_id, dt):
        pass

    def get_settlement(self, instrument, dt):
        pass

    def get_settle_price(self, order_book_id, date):
        pass

    @lru_cache(512)
    def get_bar(self, order_book_id: str, dt: date, frequency: str = '1d') -> BarObject:
        """
        Out[43]: BarObject(order_book_id=000001.XSHG, datetime=2016-06-08 00:00:00, open=2932.3761,
        close=2927.159, high=2937.9863, low=2908.3666, limit_up=nan, limit_down=nan)
        """
        pass

    def get_open_auction_bar(self, order_book_id, dt):
        """
        Out[44]: PartialBarObject(order_book_id=000001.XSHG, datetime=2016-06-08 00:00:00, open=2932.3761, limit_up=nan, limit_down=nan, last=2932.3761)
        """
        pass

    def history(self, order_book_id, bar_count, frequency, field, dt):
        """
        >>>dp.history("000001.XSHG", 3, '1d', "close", event.calendar_dt)
        Out[46]:
        2016-06-06    2934.0979
        2016-06-07    2936.0449
        2016-06-08    2927.1590
        dtype: float64
        """
        pass

    def fast_history(self, order_book_id, bar_count, frequency, field, dt):
        pass

    def history_bars(self, order_book_id, bar_count, frequency, field, dt,
                     skip_suspended=True, include_now=False,
                     adjust_type='pre', adjust_orig=None):
        """
        dp.history_bars("000001.XSHG", 10, '1d', ["close", 'open'], event.calendar_dt)
        Out[48]:
        array([(2822.4429, 2813.5431), (2821.046 , 2817.9684),
               (2822.4509, 2814.6507), (2916.616 , 2822.5927),
               (2913.5077, 2917.1541), (2925.2293, 2911.2193),
               (2938.682 , 2929.7881), (2934.0979, 2940.9943),
               (2936.0449, 2936.2821), (2927.159 , 2932.3761)],
              dtype={'names': ['close', 'open'], 'formats': ['<f8', '<f8'], 'offsets': [16, 8], 'itemsize': 56})
        """
        pass

    def history_ticks(self, order_book_id, count, dt):
        pass

    def current_snapshot(self, order_book_id, frequency, dt):
        pass

    def available_data_range(self, frequency):
        """
        dp.available_data_range('1d')
        Out[57]: (datetime.date(2005, 1, 4), datetime.date(2023, 6, 1))
        """
        return self._data_source.available_data_range(frequency)

    def get_commission_info(self, order_book_id):
        instrument = self.instruments(order_book_id)
        return self._data_source.get_commission_info(instrument)

    def get_merge_ticks(self, order_book_id_list, trading_date, last_dt=None):
        return self._data_source.get_merge_ticks(order_book_id_list, trading_date, last_dt)

    def is_suspended(self, order_book_id, dt, count=1):
        # type: (str, DateLike, int) -> Union[Sequence[bool], bool]
        if count == 1:
            return self._data_source.is_suspended(order_book_id, [dt])[0]

        trading_dates = self.get_n_trading_dates_until(dt, count)
        return self._data_source.is_suspended(order_book_id, trading_dates)

    def is_st_stock(self, order_book_id, dt, count=1):
        if count == 1:
            return self._data_source.is_st_stock(order_book_id, [dt])[0]

        trading_dates = self.get_n_trading_dates_until(dt, count)
        return self._data_source.is_st_stock(order_book_id, trading_dates)

    def get_tick_size(self, order_book_id):
        """
        0.01, 一般都是这个
        """
        return self.instruments(order_book_id).tick_size()

    def get_last_price(self, order_book_id):
        pass

    def all_instruments(self, types, dt=None):
        pass

    @lru_cache(2048)
    def instrument(self, sym_or_id):
        return next(iter(self._data_source.get_instruments(id_or_syms=[sym_or_id])), None)

    def instruments(self, sym_or_ids):
        # type: (StrOrIter) -> Union[None, Instrument, List[Instrument]]
        if isinstance(sym_or_ids, str):
            return next(iter(self._data_source.get_instruments(id_or_syms=[sym_or_ids])), None)
        else:
            return list(self._data_source.get_instruments(id_or_syms=sym_or_ids))

    def get_future_contracts(self, underlying, date):
        # type: (str, DateLike) -> List[str]
        return sorted(i.order_book_id for i in self.all_instruments(
            [INSTRUMENT_TYPE.FUTURE], date
        ) if i.underlying_symbol == underlying and not Instrument.is_future_continuous_contract(i.order_book_id))

    def get_trading_period(self, sym_or_ids, default_trading_period=None):
        pass

    def is_night_trading(self, sym_or_ids):
        # type: (StrOrIter) -> bool
        return any((instrument.trade_at_night for instrument in self.instruments(sym_or_ids)))

    



