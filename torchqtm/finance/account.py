from collections import OrderedDict
import math
import logging
import numpy as np
import pandas as pd
import typing

from torchqtm.finance.orders.order import Order
from torchqtm.finance.portfolio import Portfolio
from torchqtm.finance.positions.tracker import PositionsTracker
from typing import Literal, List, Dict, Tuple, Optional, Any
from torchqtm.finance.positions.position import Position
from torchqtm.data.data_portal import DataPortal
from torchqtm.finance.transaction import Transaction
from torchqtm.types import DATA_FREQUENCIES, ASSET_TYPE, EVENT_TYPE
from torchqtm.assets import Asset
from torchqtm.finance.orders.tracker import OrdersTracker
from torchqtm.utils.datetime_utils import DateTimeManager, DateTimeMixin
log = logging.getLogger("Account")


class AccountTracker(object):
    pass


class Account(object, DateTimeMixin):
    """
    The account trackers all get_orders and get_transactions as well as the current state of
    the portfolio and positions.
    """
    def __init__(
            self,
            datetime_manager: DateTimeManager,
            trading_sessions: pd.DatetimeIndex,
            capital_base: float,
            data_frequency: DATA_FREQUENCIES,
    ):
        if len(trading_sessions):
            start = trading_sessions[0]
        else:
            start = None

        DateTimeMixin.__init__(self, datetime_manager)

        # TODO: initialize
        self.positions_tracker: PositionsTracker = PositionsTracker(data_frequency)
        self.orders_tracker: OrdersTracker = OrdersTracker()
        self.slippage_models = self.orders_tracker.slippage_models
        self.commission_models = self.orders_tracker.commission_models

        self.cash_flow: float = 0.0
        self.starting_cash: float = capital_base
        self.portfolio_value: float = capital_base
        self.pnl: float = 0.0
        self.returns: float = 0.0
        self.cash: float = 0.0
        # 构建一个reference
        self.positions: typing.Dict = self.positions_tracker.data
        self.start_date: pd.Timestamp = start
        self.positions_value: float = 0.0
        self.positions_exposure: float = 0.0

        self._dirty_portfolio = False
        self.daily_returns_series: pd.Series = pd.Series(np.nan, index=trading_sessions)

        self._previous_total_returns: float = 0.0

        self._position_stats: Optional[Any] = None

        self._dirty_account: bool = True

        self._processed_transactions_by_dt: Dict[pd.Timestamp, Transaction] = {}

        self._orders_by_dt: Dict[pd.Timestamp, OrderedDict] = {}

        self._orders_by_id: typing.OrderedDict[str, Order] = OrderedDict()

        self._payout_last_sale_prices: Dict[Asset, float] = {}

        self._session_count: int = 0

    @property
    def todays_returns(self):
        # compute today's returns in returns space instead of portfolio-value
        # space to work even when we have capital changes
        return (self.returns + 1) / (self._previous_total_returns + 1) - 1

    def start_of_session(self):
        self._processed_transactions_by_dt.clear()
        self._orders_by_dt.clear()
        self._orders_by_id.clear()
        self._previous_total_returns = self.returns

    def end_of_bar(self, session_idx):
        self.daily_returns_series.values[session_idx] = self.todays_returns

    # TODO: 感觉会出KeyError
    def end_of_session(self, session_idx):
        self.daily_returns_series.values[session_idx] = self.todays_returns

    def sync_last_sale_prices(
            self,
            dt: pd.Timestamp,
            data_portal: DataPortal,
            handle_onn_market_minutes: bool = False,
    ):
        self.positions_tracker.sync_last_sale_price(
            dt,
            data_portal,
            handle_onn_market_minutes,
        )
        self._dirty_portfolio = True

    @staticmethod
    def _calculate_payout(multiplier, amount, old_price, price):
        return (price - old_price) * multiplier * amount

    def update_cash_flow(self, amount: float):
        self._dirty_portfolio = True
        self.cash_flow += amount
        self.cash += amount

    def handle_transaction(self, txn: Transaction):
        """Add a transaction to ledger, updating the current state as needed.

        Parameters
        ----------
        txn : Transaction
            The transaction to execute.
        """
        asset = txn.asset
        if asset.type == ASSET_TYPE.Future:
            try:
                old_price = self._payout_last_sale_prices[asset]
            except KeyError:
                self._payout_last_sale_prices[asset] = txn.price
            else:
                position = self.positions[asset]
                amount = position.amount
                price = txn.price

                self.update_cash_flow(
                    self._calculate_payout(
                        asset.price_multiplier,
                        amount,
                        old_price,
                        price,
                    ),
                )

                if amount + txn.amount == 0:
                    del self._payout_last_sale_prices[asset]
                else:
                    self._payout_last_sale_prices[asset] = price
        else:
            self.update_cash_flow(-(txn.price * txn.amount))

        self.positions_tracker.handle_transaction(txn)

        try:
            self._processed_transactions_by_dt[txn.dt].append(
                txn,
            )
        except KeyError:
            self._processed_transactions_by_dt[txn.dt] = [txn]

    def handle_splits(self, splits: List[Tuple[Asset, float]]):
        """Process a list of splits by modifying any positions as needed.

        Parameters
        ----------
        splits: List[Tuple[Asset, float]]
            A list of splits. Each split is a tuple of (asset, ratio)

        """
        leftover_cash = self.positions_tracker.handle_splits(splits)
        if leftover_cash > 0:
            self.update_cash_flow(leftover_cash)

    def handle_order(self, order: Order):
        """Keep track of an order that was placed.

        Parameters
        ----------
        order : Order
            The order to record.
        """
        try:
            dt_orders = self._orders_by_dt[order.dt]
        except KeyError:
            self._orders_by_dt[order.dt] = OrderedDict(
                [
                    (order.id, order),
                ]
            )
            self._orders_by_id[order.id] = order
        else:
            self._orders_by_id[order.id] = dt_orders[order.id] = order
            # to preserve the order of the get_orders by modified date
            OrderedDict.move_to_end(dt_orders, order.id, last=True)

        OrderedDict.move_to_end(self._orders_by_id, order.id, last=True)

    def handle_commission(self, commission):
        """Process the commission.

        Parameters
        ----------
        commission : zp.Event
            The commission being paid.
        """
        # TODO: fix this
        asset = commission["asset"]
        cost = commission["cost"]
        self.positions_tracker.handle_commission(asset, cost)
        # 为什么是-cost
        self.update_cash_flow(-cost)

    def close_position(self, asset, dt, data_portal):
        # TODO: fix this
        txn = self.positions_tracker.maybe_create_position_transaction(
            asset,
            dt,
            data_portal,
        )
        if txn is not None:
            self.handle_transaction(txn)

    def handle_dividends(self, next_session, asset_finder, adjustment_reader):
        # TODO: fix this
        pass

    def capital_change(self, change_amount: float) -> None:
        self.update_portfolio()
        self.portfolio_value += change_amount
        self.cash += change_amount

    def get_transactions(self, dt: Optional[pd.Timestamp] = None):
        """Retrieve the dict-form of all of the get_transactions in a given bar or
        for the whole simulation.

        Parameters
        ----------
        dt : pd.Timestamp or None, optional
            The particular datetime to look up get_transactions for. If not passed,
            or None is explicitly passed, all of the get_transactions will be
            returned.

        Returns
        -------
        get_transactions : List[Transaction]
            The transaction information.
        """
        if dt is None:
            # flatten the by-day get_transactions
            return [
                txn
                for by_day in self._processed_transactions_by_dt.values()
                for txn in by_day
            ]

        return self._processed_transactions_by_dt.get(dt, [])

    def get_orders(self, dt=None):
        if dt is None:
            # get_orders by id is already flattened
            return [o.to_dict() for o in self._orders_by_id.values()]

        return [o.to_dict() for o in self._orders_by_dt.get(dt, {}).values()]

    def _get_payout_total(self, positions):
        calculate_payout = self._calculate_payout
        payout_last_sale_prices = self._payout_last_sale_prices

        total = 0
        for asset, old_price in payout_last_sale_prices.items():
            position = positions[asset]
            payout_last_sale_prices[asset] = price = position.last_sale_price
            amount = position.amount
            total += calculate_payout(
                asset.price_multiplier,
                amount,
                old_price,
                price,
            )

        return total

    def update_portfolio(self):
        if not self._dirty_portfolio:
            return None

        self.positions_value = self.positions_tracker.stats.net_value
        position_value = self.positions_tracker.stats.net_value
        self.positions_exposure = self.positions_tracker.stats.net_exposure
        self.update_cash_flow(self._get_payout_total(self.positions_tracker.data))

        start_value = self.portfolio_value

        # update the new starting value
        self.portfolio_value = self.cash + position_value
        end_value = self.cash + position_value

        pnl = end_value - start_value
        if start_value != 0:
            returns = pnl / start_value
        else:
            returns = 0.0

        self.pnl += pnl
        self.returns = (1 + self.returns) * (1 + returns) - 1

        # the portfolio has been fully synced
        self._dirty_portfolio = False

    def subscribe_cash(self, amount: float):
        log.warning("有外部资金流入, 风险测度可能不再准确")
        self.update_cash_flow(amount)

    def get_portfolio(self):
        """Compute the current portfolio.

        Notes
        -----
        This is cached, repeated access will not recompute the portfolio until
        the portfolio may have changed.
        """
        # 这么搞一波相当于是每次调用portfolio的时候都会更新一次
        self.update_portfolio()
        return self._portfolio

    def calculate_period_stats(self):
        if self.portfolio_value == 0:
            gross_leverage = net_leverage = np.inf
        else:
            gross_leverage = self.positions_tracker.stats.gross_exposure / self.portfolio_value
            net_leverage = self.positions_tracker.stats.net_exposure / self.portfolio_value
        return self.portfolio_value, gross_leverage, net_leverage

    # 提供orders_tracker的封装
    def set_dt(self, dt):
        return self.orders_tracker.set_date(dt)

    def order(
            self,
            asset: Asset,
            amount: int,
            stop: float = None,
            limit: float = None,
            order_id: str = None,
    ) -> Optional[str]:
        return self.orders_tracker.order(asset, amount, stop, limit, order_id)

    def cancel(
            self,
            order_id: str,
            relay_status: bool = True,
    ) -> None:
        return self.orders_tracker.cancel(order_id, relay_status)

    def cancel_all_orders_for_asset(
            self,
            asset: Asset,
            warn: bool = False,
            relay_status: bool = True,
    ) -> None:
        return self.orders_tracker.cancel_all_orders_for_asset(asset, warn, relay_status)

    def execute_daily_cancel_policy(self, event: EVENT_TYPE) -> None:
        return self.orders_tracker.execute_daily_cancel_policy(event)

    def execute_cancel_policy(self, event: EVENT_TYPE) -> None:
        return self.orders_tracker.execute_cancel_policy(event)























