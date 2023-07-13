# blotter is the abstraction for the collection of get_orders
# 用于控制订单的整体行为

from abc import ABCMeta, abstractmethod
from torchqtm.finance.orders.order import Order
from collections import defaultdict
import pandas as pd
import logging
from typing import Dict, List, Optional, DefaultDict, Union, Any, Type, Tuple
from copy import copy

from torchqtm.finance.models.commission import CommissionModel, EquityCommissionModel, FutureCommissionModel
from torchqtm.finance.models.slippage import SlippageModel, EquitySlippageModel, FutureSlippageModel
from torchqtm.finance.models.cancel_policy import CancelPolicy, NeverCancel
from torchqtm.assets import Asset, Equity, Future
from torchqtm.data.data_portal import DataPortal
from torchqtm.finance.transaction import Transaction
from torchqtm.finance.models.commission import (
    # DEFAULT_PER_CONTRACT_COST,
    # FUTURE_EXCHANGE_FEES_BY_SYMBOL,
    PerContract,
    PerShare,
)
from torchqtm.finance.models.slippage import (
    DEFAULT_FUTURE_VOLUME_SLIPPAGE_BAR_LIMIT,
    DEFAULT_PER_CONTRACT_COST,
    FUTURE_EXCHANGE_FEES_BY_SYMBOL,
    VolatilityVolumeShare,
    FixedBasisPointsSlippage,
)

from torchqtm.types import ASSET_TYPE, EVENT_TYPE
from torchqtm.utils.datetime_utils import DateTimeManager, DateTimeMixin

log = logging.getLogger("Blotter")
warning_logger = logging.getLogger("AlgoWarning")


class OrdersTracker(DateTimeMixin):
    def __init__(
            self,
            datetime_manager: DateTimeManager = None,
            equity_slippage: SlippageModel = None,
            future_slippage: SlippageModel = None,
            equity_commission: CommissionModel = None,
            future_commission: CommissionModel = None,
            cancel_policy: CancelPolicy = None,
    ):

        DateTimeMixin.__init__(self, datetime_manager)
        self.cancel_policy = cancel_policy

        # these get_orders are aggregated by asset
        self.open_orders: DefaultDict[Asset, List[Order]] = defaultdict(list)

        # keep a dict of get_orders by their own id
        self.data: Dict[str, Order] = {}

        # holding get_orders that have come in since the last event.
        self.new_orders: List[Order] = []
        self.max_shares = int(1e11)
        # Here we use type to indicate that the key is Asset not the instance of Asset
        # TODO
        self.slippage_models: Dict[ASSET_TYPE, SlippageModel] = {
            ASSET_TYPE.Equity: equity_slippage or FixedBasisPointsSlippage(),
            ASSET_TYPE.Future: future_slippage
                               or VolatilityVolumeShare(
                volume_limit=DEFAULT_FUTURE_VOLUME_SLIPPAGE_BAR_LIMIT,
            ),
        }
        self.commission_models: Dict[ASSET_TYPE, CommissionModel] = {
            ASSET_TYPE.Equity: equity_commission or PerShare(),
            ASSET_TYPE.Future: future_commission
                               or PerContract(
                cost=DEFAULT_PER_CONTRACT_COST,
                exchange_fee=FUTURE_EXCHANGE_FEES_BY_SYMBOL,
            ),
        }

    def __repr__(self):
        raw_expression = f"""
{self.__class__.__name__}(
    slippage_models={self.slippage_models},
    commission_models={self.commission_models},
    open_orders={self.open_orders},
    get_orders={self.data},
    new_orders={self.new_orders},
    current_dt={self.current_dt}
)
        """.strip()
        return raw_expression

    def order(
            self,
            asset: Asset,
            amount: int,
            stop: float = None,
            limit: float = None,
            order_id: str = None,
    ) -> Optional[str]:

        if amount == 0:
            return None
        elif amount > self.max_shares:
            raise OverflowError("Can't order more than %d shares" % self.max_shares)

        order = Order(
            dt=self.current_dt,
            asset=asset,
            amount=amount,
            stop=stop,
            limit=limit,
            id=order_id,
        )

        self.open_orders[order.asset].append(order)
        self.data[order.id] = order
        self.new_orders.append(order)
        return order.id

    def cancel(
            self,
            order_id: str,
            relay_status: bool = True,
    ) -> None:
        if order_id not in self.data:
            return None

        current_order = self.data[order_id]

        if current_order.open:
            order_list = self.open_orders[current_order.asset]
            if current_order in order_list:
                order_list.remove(current_order)

            if current_order in self.new_orders:
                self.new_orders.remove(current_order)
            current_order.cancel()  # change the status to canceled
            current_order.dt = self.current_dt

            if relay_status:
                self.new_orders.append(current_order)

    def cancel_all_orders_for_asset(
            self,
            asset: Asset,
            warn: bool = False,
            relay_status: bool = True,
    ) -> None:
        """
        Cancel all open get_orders for a given asset
        """
        orders = self.open_orders[asset]
        for order in orders:
            if warn:
                # Message appropriately depending on whether there's
                # been a partial fill or not.
                if order.filled > 0:
                    warning_logger.warning(
                        "Your order for {order_amt} shares of "
                        "{order_sym} has been partially filled. "
                        "{order_filled} shares were successfully "
                        "purchased. {order_failed} shares were not "
                        "filled by the end of day and "
                        "were canceled.".format(
                            order_amt=order.amount,
                            order_sym=order.asset.symbol,
                            order_filled=order.filled,
                            order_failed=order.amount - order.filled,
                        )
                    )
                elif order.filled < 0:
                    warning_logger.warning(
                        "Your order for {order_amt} shares of "
                        "{order_sym} has been partially filled. "
                        "{order_filled} shares were successfully "
                        "sold. {order_failed} shares were not "
                        "filled by the end of day and "
                        "were canceled.".format(
                            order_amt=order.amount,
                            order_sym=order.asset.symbol,
                            order_filled=-1 * order.filled,
                            order_failed=-1 * (order.amount - order.filled),
                        )
                    )
                else:
                    warning_logger.warning(
                        "Your order for {order_amt} shares of "
                        "{order_sym} failed to fill by the end of day "
                        "and was canceled.".format(
                            order_amt=order.amount,
                            order_sym=order.asset.symbol,
                        )
                    )

        assert not orders
        del self.open_orders[asset]

    # End of day cancel for daily frequency
    def execute_daily_cancel_policy(self, event: EVENT_TYPE) -> None:
        if self.cancel_policy.should_cancel(event):
            warn = self.cancel_policy.warn_on_cancel
            for asset in copy(self.open_orders):
                orders = self.open_orders[asset]
                if len(orders) > 1:
                    # 为什么只取消了第0个订单
                    order = orders[0]
                    self.cancel(order.id, relay_status=True)
                    if warn:
                        if order.filled > 0:
                            warning_logger.warning(
                                "Your order for {order_amt} shares of "
                                "{order_sym} has been partially filled. "
                                "{order_filled} shares were successfully "
                                "purchased. {order_failed} shares were not "
                                "filled by the end of day and "
                                "were canceled.".format(
                                    order_amt=order.amount,
                                    order_sym=order.asset.symbol,
                                    order_filled=order.filled,
                                    order_failed=order.amount - order.filled,
                                )
                            )
                        elif order.filled < 0:
                            warning_logger.warning(
                                "Your order for {order_amt} shares of "
                                "{order_sym} has been partially filled. "
                                "{order_filled} shares were successfully "
                                "sold. {order_failed} shares were not "
                                "filled by the end of day and "
                                "were canceled.".format(
                                    order_amt=order.amount,
                                    order_sym=order.asset.symbol,
                                    order_filled=-1 * order.filled,
                                    order_failed=-1 * (order.amount - order.filled),
                                )
                            )
                        else:
                            warning_logger.warning(
                                "Your order for {order_amt} shares of "
                                "{order_sym} failed to fill by the end of day "
                                "and was canceled.".format(
                                    order_amt=order.amount,
                                    order_sym=order.asset.symbol,
                                )
                            )

    def execute_cancel_policy(self, event: EVENT_TYPE) -> None:
        if self.cancel_policy.should_cancel(event):
            warn = self.cancel_policy.warn_on_cancel
            for asset in copy(self.open_orders):
                self.cancel_all_orders_for_asset(asset, warn, relay_status=False)

    def reject(self, order_id: str, reason: str = "") -> None:
        """
        Mark the given order as 'rejected', which is functionally similar to
        cancelled. The distinction is that rejections are involuntary (and
        usually include a message from a broker indicating why the order was
        rejected) while cancels are typically user-driven.
        """
        if order_id not in self.data:
            return None

        current_order = self.data[order_id]
        order_list = self.open_orders[current_order.asset]
        if current_order in order_list:
            order_list.remove(current_order)

        if current_order in self.new_orders:
            self.new_orders.remove(current_order)

        current_order.reject(reason=reason)
        current_order.dt = self.current_dt

        self.new_orders.append(current_order)

    def hold(self, order_id: str, reason: str = "") -> None:
        """
        Mark the order with order_id as 'held'. Held is functionally similar
        to 'open'. When a fill (full or partial) arrives, the status
        will automatically change back to open/filled as necessary.
        """
        if order_id not in self.data:
            return None

        cur_order = self.data[order_id]
        if cur_order.open:
            if cur_order in self.new_orders:
                self.new_orders.remove(cur_order)
            cur_order.hold(reason=reason)
            cur_order.dt = self.current_dt
            # we want this order's new status to be relayed out
            # along with newly placed get_orders.
            self.new_orders.append(cur_order)

    def handle_splits(self, splits: List[Tuple[Asset, float]]) -> None:
        """
        Parameters
        ----------
        splits: list
            a list of splits. Each split is a tuple of (asset, ratio)
        """
        for asset, ratio in splits:
            if asset not in self.open_orders:
                continue

            orders_to_modify = self.open_orders[asset]
            for order in orders_to_modify:
                order.handle_split(ratio)

    def get_transactions(self, data: DataPortal) -> Tuple[List, List, List]:
        """Create a list of get_transactions based on the current open get_orders,
        slippage models, and commission models.

        Returns
        -------
        closed_orders: list
        commissions_list: list
        transactions_list: list
        """
        closed_orders: List[Order] = []
        commissions: List[Dict[str, Union[Asset, Order, float]]] = []
        transactions: List[Transaction] = []

        if self.open_orders:
            for asset, asset_orders in self.open_orders.items():
                slippage_model = self.slippage_models[asset.type]

                # 注意一下, 这里的order是asset_orders中的一个reference, 所以order状态的改变将直接体现在asset_orders中
                try:
                    it = iter(slippage_model.simulate(data, asset, asset_orders))
                except TypeError:
                    return transactions, commissions, closed_orders
                while True:
                    try:
                        order, txn = next(it)
                        commission_model = self.commission_models[asset.type]
                        additional_commission = commission_model.calculate(order, txn)

                        if additional_commission > 0:
                            commissions.append(
                                {
                                    "asset": order.asset,
                                    "order": order,
                                    "cost": additional_commission
                                }
                            )

                        order.filled += txn.amount
                        order.commission += additional_commission

                        order.dt = txn.dt

                        transactions.append(txn)

                        if not order.open:
                            closed_orders.append(order)
                    except StopIteration:
                        break
        return transactions, commissions, closed_orders

    def get_new_orders(self):
        return self.new_orders

    def sweep_new_orders(self):
        self.new_orders = []

    def prune_orders(self, closed_orders: List[Order]) -> None:
        """
        Remove all given get_orders from the blotter's open_orders list
        """
        for order in closed_orders:
            asset = order.asset
            asset_orders = self.open_orders[asset]
            try:
                asset_orders.remove(order)
            except ValueError:
                continue

        for asset in list(self.open_orders.keys()):
            if len(self.open_orders[asset]) == 0:
                del self.open_orders[asset]
