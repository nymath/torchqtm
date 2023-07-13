from abc import ABCMeta, abstractmethod
import datetime
import logging
import pandas as pd
import numpy as np

from torchqtm.edbt.sim_params import SimulationParameters
from torchqtm.finance.metrics.tracker import MetricsTracker
from torchqtm.finance.orders.order import Order
from torchqtm.finance.orders.tracker import OrdersTracker
from torchqtm.assets import Asset, Equity, Future
from torchqtm.data.data_portal import DataPortal
from torchqtm.edbt.gens.sim_engine import DailySimulationClock, MinuteSimulationClock
import typing
from torchqtm.types import EVENT_TYPE, ASSET_TYPE
from torchqtm.finance.account import Account
from torchqtm.utils.exchange_calendar import XSHGExchangeCalendar
from torchqtm.utils.datetime_utils import DateTimeManager, DateTimeMixin


class TradingAlgorithm(DateTimeMixin):
    _allow_setattr: bool = False

    def __init__(
            self,
            sim_params: SimulationParameters,
            data_portal: DataPortal,
            namespace=None,
            trading_calendar: XSHGExchangeCalendar = None,
            datetime_manager: DateTimeManager = None,
            benchmark_returns=None,
            account: Account = None,
            metrics_tracker: MetricsTracker = None,
    ):
        self._allow_setattr = True
        # Create a reference to the data_portal datetime
        DateTimeMixin.__init__(self, datetime_manager)
        assert account is not None
        assert metrics_tracker is not None

        self.data_portal = data_portal

        self.benchmark_returns = benchmark_returns

        self.sim_params = sim_params

        self.trading_calendar = trading_calendar

        self._last_sync_time = None
        self.metrics_tracker = metrics_tracker

        # if self._metrics_set is None:
        #     self._metrics_set = load_metrics_set("default")

        # TODO: initialize this, 我认为不需要创建account, 只需要创建account的引用就行了
        self.account = account

        self.initialized = False

        self.capital_changes = {}
        self.recorded_vars = {}

        # self.restrictions = NoRestrictions()
        self._allow_setattr = False

    # def __setattr__(self, key, value):
    #     if self._allow_setattr or hasattr(self, key):
    #         # 思考一下用super().__setattr__好还是setattr好, 后者可能会造成递归
    #         super().__setattr__(key, value)
    #     else:
    #         raise AttributeError(f"You cannot add attributes to {self.__class__.__name__}")

    def safe_set_attr(self, attr, value):
        """
        Safely sets an attribute to the object.

        This method checks whether an attribute with the same name already exists on the object.
        If it does, it raises an AttributeError. Otherwise, it sets the attribute to the given value.

        Parameters:
            attr (str): The name of the attribute to set.
            value: The value to set the attribute to.

        Raises:
            AttributeError: If an attribute with the same name already exists on the object.
        """
        if hasattr(self, attr):
            raise AttributeError(f"Attribute {attr} already exists and cannot be set")
        else:
            self._allow_setattr = True
            try:
                setattr(self, attr, value)
            finally:
                self._allow_setattr = False

    @abstractmethod
    def initialize(self):
        """
        Set the initial params of backtest.

        Note:
            use self.safe_set_attr('i', 0) instead of self.i = 0 to avoid overwriting the existing attributes
        """
        raise NotImplementedError("should be implemented in the derived class")

    def _initialize(self):
        self.initialize()

    def before_trading_start(self):
        # self.compute_eager_pipelines()
        pass

    def _before_trading_start(self):
        """
        总体上与api保持一致, 但我用来控制一些内部行为了
        """
        self.before_trading_start()

    def handle_data(self):
        pass

    def _handle_data(self):
        # self.rule.should_trigger(self.current_dt) # FIXME
        self.handle_data()

    def analyze(self, perf):
        pass

    def _analyze(self, perf):
        self.analyze(perf)

    # def __repr__(self):
    #     pass

    def _create_sim_engine(self):
        self.on_dt_changed(self.sim_params.start_session)

        if not self.initialized:
            # TODO: do something
            self.initialized = True

        # benchmark_source = self._create_benchmark_source()

        return DailySimulationClock(
            sessions=self.sim_params.sessions,
            market_open=self.trading_calendar.market_open_time,
            market_close=self.trading_calendar.market_close_time,
        )

    def _create_benchmark_source(self):
        return None

    def _create_metrics_tracker(self):
        return None

    def _create_generator(self, sim_params):

        sim_engine = self._create_sim_engine()

        # benchmark_source = self._create_benchmark_source()

        algo_iter = AlgorithmIterator(
            algo=self,
            datetime_manager=self.datetime_manager,
            sim_params=self.sim_params,
            data_portal=self.data_portal,
            sim_engine=sim_engine,
            benchmark_source=None, # TODO: fixme
            restrictions=None,
        )
        return algo_iter
        # metrics_tracker.handle_start_of_simulation(benchmark_source)

    # def compute_eager_pipelines(self):
    #     raise NotImplementedError

    def get_generator(self):
        self._initialize()
        return self._create_generator(self.sim_params)

    def _create_daily_stats(self):
        pass

    def calculate_capital_changes(
            self,
            dt: pd.Timestamp,
            emission_rate: typing.Literal["daily", "minute"],
            is_interday: bool,
            portfolio_value_adjustment: float = 0.0,
    ):
        """
        在session start的时候调用, 计算资本利得
        """
        try:
            capital_change = self.capital_changes[dt]
        except KeyError:
            return

        self._sync_last_sale_prices()


    def run(self):
        # try:
        #     perfs = []
        #     it = self.get_generator()
        # while True:
        #     try:
        #         perf = next(it)
        #         perfs.append(perf)
        #     except StopIteration:
        #         break
        #
        # daily_stats = self._create_daily_stats(perfs)
        # self._analyze(daily_stats)
        # return daily_stats
        pass

    def record(self):
        """
        Track and record values each day
        """
        pass

    def symbol(self):
        pass

    def symbols(self):
        pass

    def _can_order_asset(self, asset) -> bool:
        pass


    #TODO: revise it
    @staticmethod
    def round(amount: float) -> int:
        return int(amount)

    def validate_order_params(self, asset, amount, limit_price, stop_price):
        # for control in self.trading_controls:
        #     for control in self.trading_controls:
        #         control.validate(
        #             asset,
        #             amount,
        #             self.portfolio,
        #             self.get_datetime(),
        #             self.trading_client.current_data,
        #         )
        # return None
        return None

    def _calculate_order_value_amount(self, asset, value):
        # """Calculates how many shares/contracts to order based on the type of
        # asset being ordered.
        # """
        # # Make sure the asset exists, and that there is a last price for it.
        # # FIXME: we should use BarData's can_trade logic here, but I haven't
        # # yet found a good way to do that.
        # normalized_date = self.trading_calendar.minute_to_session(self.datetime)
        #
        # if normalized_date < asset.start_date:
        #     raise CannotOrderDelistedAsset(
        #         msg="Cannot order {0}, as it started trading on"
        #         " {1}.".format(asset.symbol, asset.start_date)
        #     )
        # elif normalized_date > asset.end_date:
        #     raise CannotOrderDelistedAsset(
        #         msg="Cannot order {0}, as it stopped trading on"
        #         " {1}.".format(asset.symbol, asset.end_date)
        #     )
        # else:
        #     last_price = self.trading_client.current_data.current(asset, "price")
        #
        #     if np.isnan(last_price):
        #         raise CannotOrderDelistedAsset(
        #             msg="Cannot order {0} on {1} as there is no last "
        #             "price for the security.".format(asset.symbol, self.datetime)
        #         )
        #
        # if tolerant_equals(last_price, 0):
        #     zero_message = "Price of 0 for {psid}; can't infer value".format(psid=asset)
        #     if self.logger:
        #         self.logger.debug(zero_message)
        #     # Don't place any order
        #     return 0
        #
        # value_multiplier = asset.price_multiplier
        #
        # return value / (last_price * value_multiplier)
        pass

    def order(self, asset, amount, limit_price=None, stop_price=None):
        self.validate_order_params(asset, amount, limit_price, stop_price)
        self.account.order(asset, amount, limit_price, stop_price)

    def order_value(self, asset, value, limit_price=None, stop_price=None):
        if not self._can_order_asset(asset):
            return None
        amount = self._calculate_order_value_amount(asset, value)
        return self.order(asset, amount, limit_price, stop_price)

    def order_percent(self, asset, percent, limit_price=None, stop_price=None):
        pass

    def order_target(self, asset, target, limit_price=None, stop_price=None):
        pass

    def order_target_value(self, asset, target_value, limit_price=None, stop_price=None):
        pass

    def order_target_percent(self, asset, target_percent, limit_price=None, stop_price=None):
        pass

    def _sync_last_sale_prices(self, dt=None):
        if dt is None:
            dt = self.current_dt

        if dt != self._last_sync_time:
            self.account.sync_last_sale_price(
                dt,
                self.data_portal,
            )
            self._last_sync_time = dt

    def on_dt_changed(self, dt: pd.Timestamp):
        self.datetime_manager.set_dt(dt)
        self.account.set_dt(dt)

    def set_slippage(self, equity_model=None, future_model=None):
        if self.initialized:
            raise ValueError("模型已经初始化")

        if equity_model is not None:
            self.account.slippage_models[ASSET_TYPE.Equity] = equity_model

        if future_model is not None:
            self.account.slippage_models[ASSET_TYPE.Future] = future_model

    def set_commission(self, equity_model=None, future_model=None):
        if self.initialized:
            raise ValueError("模型已经初始化")

        if equity_model is not None:
            self.account.commission_models[ASSET_TYPE.Equity] = equity_model

        if future_model is not None:
            self.account.commission_models[ASSET_TYPE.Future] = future_model

    def set_data_frequency(self, value):
        pass

    @property
    def data_frequency(self):
        pass

    def get_open_orders(self, asset=None):
        """
        Retrieve all of the current open get_orders
        """
        if asset is None:
            return {
                key: [order for order in orders]
                for key, orders in self.account.orders_tracker.open_orders.items()
                if orders
            }
        if asset in self.account.orders_tracker.open_orders:
            orders = self.account.orders_tracker.open_orders[asset]
            return [order for order in orders]
        return []

    def get_order(self, order_id):
        pass

    def cancel_order(self, order_param):
        """
        Parameters
        ----------
        order_param: order_id or Order object
        """
        order_id = order_param
        if isinstance(order_param, Order):
            order_id = order_param.id
        self.account.cancel(order_id)

    # the interface of data_portal
    def history(self, symbol, field, bar_count, frequency):
        self.data_portal.get_history_window(symbol, field, bar_count, frequency)

    def __repr__(self):
        template = """
{class_name}(
capital_base={capital_base},
sim_params={sim_params},
initialized={initialized},
slippage_models={slippage_models},
commission_models={commission_models},
account={account}
)
        """.strip()
        return template.format(
            class_name=self.__class__.__name__,
            capital_base=self.sim_params.capital_base,
            sim_params=repr(self.sim_params),
            initialized=self.initialized,
            slippage_models=repr(self.account.slippage_models),
            commission_models=repr(self.account.commission_models),
            account=repr(self.account),
        )


class AlgorithmIterator(DateTimeMixin):
    """
    AlgoIterator并不是一个迭代器, 而是Iterable, 你可以把他看作sim_engine的增强
    """
    EMISSION_TO_PERF_KEY_MAP = {"minute": "minute_perf", "daily": "daily_perf"}

    def __init__(
            self,
            algo: TradingAlgorithm,
            datetime_manager: DateTimeManager,
            sim_params: SimulationParameters,
            data_portal: DataPortal,
            sim_engine: typing.Union[DailySimulationClock, MinuteSimulationClock],
            benchmark_source=0,
            restrictions=None,
    ):
        self.algo = algo
        DateTimeMixin.__init__(self, datetime_manager)
        self.data_portal = data_portal
        self.current_data = data_portal
        self.restrictions = restrictions
        self.sim_params = sim_params
        self.sim_engine = sim_engine
        self.benchmark_source = benchmark_source

    def on_bar(self, dt: pd.Timestamp):
        # 调用的时间是15:00:00, 今晚下单, 明晚成交
        self.set_dt(dt)
        self.algo.account.update_ledger()
        # self.algo.on_dt_changed(dt)

        blotter = self.algo.account.orders_tracker
        new_transactions, new_commissions, closed_orders = blotter.get_transactions(self.current_data)
        blotter.prune_orders(closed_orders)

        for transaction in new_transactions:
            self.algo.account.handle_transaction(transaction)
            order = blotter.data[transaction.order_id]
            self.algo.account.handle_order(order)  # process order仅仅只是把订单放到对应的数据结构中

        for commission in new_commissions:
            self.algo.account.handle_commission(commission)

        self.algo._handle_data()  # 这里可能会下单了

        new_orders = blotter.get_new_orders()
        blotter.sweep_new_orders()

        # 将new_order放入对应的数据结构中
        for new_order in new_orders:
            self.algo.account.handle_order(new_order)

    def on_session_start(self, dt: pd.Timestamp):
        """
        """
        # 在半夜调用这个 00:00:00
        self.set_dt(dt)
        # self.algo.on_dt_changed(dt)

        self.algo.metrics_tracker.handle_start_of_session(
            dt,
            self.algo.data_portal,
        )

    def on_session_end(self, dt):
        """
        执行订单取消政策
        """
        # execute_order_cancellation_policy()
        # algo.validate_account_controls()
        # TODO: 这里更新了账户信息, 想想怎么分离
        perf_message = self.algo.metrics_tracker.handle_end_of_session(
            dt,
            self.data_portal,
        )
        perf_message["daily_perf"]["recorded_vars"] = self.algo.recorded_vars
        return perf_message

    def on_before_trading_start(self, dt):
        self.set_dt(dt)
        # self.algo.on_dt_changed(dt)
        self.algo._before_trading_start()

    def on_exit(self):
        """
        Remove references to algo, data portal, et al to break cycles and ensure
        deterministic cleanup of these objects when the simulation finishes.
        """
        self.algo = None
        self.benchmark_source = None
        self.current_data = None
        self.data_portal = None

    def __iter__(self):

        if self.algo.data_frequency == "minute":
            raise NotImplementedError
            # def execute_order_cancellation_policy():
            #     self.algo.blotter.execute_cancel_policy(SESSION_END)
            #
            # def calculate_minute_capital_changes(dt):
            #     # process any capital changes that came between the last
            #     # and current minutes
            #     return self.algo.calculate_capital_changes(
            #         dt, emission_rate=self.emission_rate, is_interday=False
            #     )

        elif self.algo.data_frequency == "daily":

            def execute_order_cancellation_policy():
                # self.algo.account.execute_daily_cancel_policy(EVENT_TYPE.SESSION_END)
                # TODO: implement it
                return None
            def calculate_minute_capital_changes(dt):
                return []

        else:

            def execute_order_cancellation_policy():
                pass

            def calculate_minute_capital_changes(dt):
                return []

        event_bus = iter(self.sim_engine)
        while True:
            try:
                dt, event = next(event_bus)  # 想想能不能改进为一个队列
                if event == EVENT_TYPE.BAR:  # Bar event 产生于每天收盘, 也就是15:00:00
                    self.on_bar(dt)

                elif event == EVENT_TYPE.SESSION_START:
                    self.on_session_start(dt)

                elif event == EVENT_TYPE.SESSION_END:
                    positions = self.algo.account.positions
                    execute_order_cancellation_policy()
                    # TODO:
                    # self.algo.validate_account_controls()
                    yield self._get_daily_message(dt, self.algo, self.algo.metrics_tracker)

                elif event == EVENT_TYPE.MINUTE_START:
                    raise NotImplementedError

                elif event == EVENT_TYPE.MINUTE_END:
                    raise NotImplementedError

                elif event == EVENT_TYPE.BEFORE_TRADING_START:
                    self.on_before_trading_start(dt)

            except StopIteration:
                break

    def _cleanup_expired_assets(self, dt, position_assets):
        """
        Clear out any assets that have expired before starting a new sim day.

        Performs two functions:

        1. Find all assets for which we have open get_orders and clears any get_orders whose aasets are
        on or after their auto_close_date.

        2. Find all assets for which we have positions and generates close_position events for any assets that have
        reached their auto_close_date.
        """
        pass

    def _get_daily_message(self, dt, algo, metrics_tracker):
        """
        Get a perf message for the given datetime.
        """
        perf_message = metrics_tracker.handle_end_of_session(dt, self.data_portal)
        perf_message["daily_perf"]["recorded_vars"] = algo.recorded_vars
        return perf_message

    def _get_minute_message(self, dt, algo, metrics_tracker):
        """
        Get a perf message for the given datetime
        """
        pass


class TestAlgo(TradingAlgorithm):
    def initialize(self):
        pass

    def before_trading_start(self):
        pass

    def handle_data(self):
        self.history("000001.XSHE", "close", 10, "1d")

    def analyze(self):
        pass


if __name__ == "__main__":
    test = TestAlgo()

