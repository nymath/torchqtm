from torchqtm.utils.exchange_calendar import XSHGExchangeCalendar, DateIterator


class BarData:
    """
    DataPortal的外部封装
    """
    def __init__(
            self,
            data_portal,
            simulation_dt_func,
            data_frequency,
            trading_calendar: XSHGExchangeCalendar,
            restrictions,
    ):
        self.data_portal = data_portal
        self.simulation_dt_func = simulation_dt_func
        self.data_frequency = data_frequency
        self._views = {}
        self._daily_mode = (self.data_frequency == "daily")
        self._adjust_minutes = False
        self._trading_calendar = trading_calendar
        self._is_restricted = restrictions.is_restricted
        self._date_iterator = DateIterator(start_date=trading_calendar.start_date)

    def update(self):
        self._date_iterator.update()

    @property
    def current_session(self):
        return self._date_iterator.current_session

    def _get_current_minute(self):
        pass

    def current(self, asset, filed):
        return self._

    def current_chain(self, continues_future):
        pass

    def can_trade(self, assets):
        pass

    def _can_trade_for_asset(self, asset, dt, adjusted_dt, data_portal):
        pass

    def is_stale(self, assets):
        pass

    def _is_stale_for_asset(self, asset, dt, adjusted_dt, data_portal):
        pass

    def history(self, assets, fields, bar_count, frequency):
        pass



    



