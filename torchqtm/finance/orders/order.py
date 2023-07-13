import uuid
import math
from enum import Enum
from sys import float_info
import pandas as pd
from typing import Dict, Tuple, Optional
from torchqtm.assets import Asset
from torchqtm.types import ORDER_STATUS, ORDER_TYPE
import torchqtm.finance.zp_math as zp_math

SELL = 1 << 0
BUY = 1 << 1
STOP = 1 << 2
LIMIT = 1 << 3

ORDER_FIELDS_TO_IGNORE = {"type", "direction", "_status", "asset"}


def consistent_round(val):
    if (val % 1) >= 0.5:
        return math.ceil(val)
    else:
        return round(val)


def asymmetric_round_price(price, prefer_round_down, tick_size, diff=0.95):
    precision = zp_math.number_of_decimal_places(tick_size)
    multiplier = int(tick_size * (10**precision))
    diff -= 0.5  # shift the difference down
    diff *= 10**-precision  # adjust diff to precision of tick size
    diff *= multiplier  # adjust diff to value of tick_size

    # Subtracting an epsilon from diff to enforce the open-ness of the upper
    # bound on buys and the lower bound on sells.  Using the actual system
    # epsilon doesn't quite get there, so use a slightly less epsilon-ey value.
    epsilon = float_info.epsilon * 10
    diff = diff - epsilon

    # relies on rounding half away from zero, unlike numpy's bankers' rounding
    rounded = tick_size * consistent_round(
        (price - (diff if prefer_round_down else -diff)) / tick_size
    )
    if zp_math.tolerant_equals(rounded, 0.0):
        return 0.0
    return rounded


# TODO: create a base class for Order
class Order:
    # using __slot__ to save on memory usage.
    __slots__ = [
        "id",
        "dt",
        "reason",
        "created",
        "asset",
        "amount",
        "filled",
        "commission",
        "_status",
        "stop",
        "limit",
        "stop_reached",
        "limit_reached",
        "direction",
        "type",
        "broker_order_id",
    ]

    def __init__(
            self,
            dt: pd.Timestamp,
            asset: Asset,
            amount: int,
            stop: Optional[float] = None,
            limit: Optional[float] = None,
            filled: int = 0,
            commission: float = .0,
            id: Optional[str] = None,
    ):
        self.id = self.make_id() if id is None else id
        self.dt = dt
        self.reason = None
        self.created = dt
        self.asset = asset
        self.amount = amount
        self.filled = filled
        self.commission = commission
        self._status = ORDER_STATUS.OPEN
        self.type = self.get_order_type(stop, limit)
        # then, we need to modify the stop, limit price
        self.stop, self.limit = self.get_stop_limit_price(stop, limit, self.amount > 0)
        self.stop_reached = False
        self.limit_reached = False
        self.direction = math.copysign(1, self.amount)
        self.broker_order_id = None

    @staticmethod
    def make_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def get_order_type(stop_price: Optional[float], limit_price: Optional[float]):
        if limit_price and stop_price:
            return ORDER_TYPE.STOP_LIMIT
        elif not limit_price and stop_price:
            return ORDER_TYPE.STOP
        elif limit_price and not stop_price:
            return ORDER_TYPE.LIMIT
        else:
            return ORDER_TYPE.MARKET

    @staticmethod
    def get_stop_limit_price(
            stop: Optional[float],
            limit: Optional[float],
            is_buy: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        if not limit and not stop:
            # for market order
            true_stop = None
            true_limit = None
        elif limit and not stop:
            # for limit order
            true_stop = None
            true_limit = asymmetric_round_price(limit, is_buy, tick_size=0.01)
        elif not limit and stop:
            # for stop order
            # why here we should use not is_buy
            true_stop = asymmetric_round_price(stop, not is_buy, tick_size=0.01)
            true_limit = None
        else:
            # for stop limit order
            true_stop = asymmetric_round_price(stop, not is_buy, tick_size=0.01)
            true_limit = asymmetric_round_price(limit, is_buy, tick_size=0.01)

        return true_stop, true_limit

    def to_dict(self) -> Dict[str, object]:
        dct = {
            name: getattr(self, name)
            for name in self.__slots__
            if name not in ORDER_FIELDS_TO_IGNORE
        }

        if self.broker_order_id is None:
            del dct["broker_order_id"]

        dct['status'] = self.status
        return dct

    def check_trigger(self, price: float, dt: pd.Timestamp) -> None:
        """
        handle_transaction internal state based on price triggers and the trade event's price.
        """
        stop_reached, limit_reached, sl_stop_reached = self.check_order_triggers(price)
        if (stop_reached, limit_reached) != (self.stop_reached, self.limit_reached):
            self.dt = dt
        self.stop_reached = stop_reached
        self.limit_reached = limit_reached
        if sl_stop_reached:
            # Change the STOP LIMIT order into a LIMIT order
            self.stop = None

    def check_order_triggers(self, current_price: float) -> Tuple[bool, bool, bool]:
        """
        return (stop_reached, limit_reached)
        - `market order`: (False, False)
        - `stop order`: (~, False)
        - `limit order`: (False, ~)
        - `stop limit order`: (~, ~)
        """
        if self.triggered:
            return self.stop_reached, self.limit_reached, False

        stop_reached = False
        limit_reached = False
        sl_stop_reached = False

        order_type = 0

        if self.amount > 0:
            order_type |= BUY
        else:
            order_type |= SELL

        if self.stop is not None:
            order_type |= STOP

        if self.limit is not None:
            order_type |= LIMIT

        if order_type == BUY | STOP | LIMIT:
            if current_price >= self.stop:
                sl_stop_reached = True
                if current_price <= self.limit:
                    limit_reached = True
        elif order_type == SELL | STOP | LIMIT:
            if current_price <= self.stop:
                sl_stop_reached = True
                if current_price >= self.limit:
                    limit_reached = True
        elif order_type == BUY | STOP:
            if current_price >= self.stop:
                stop_reached = True
        elif order_type == SELL | STOP:
            if current_price <= self.stop:
                stop_reached = True
        elif order_type == BUY | LIMIT:
            if current_price <= self.limit:
                limit_reached = True
        elif order_type == SELL | LIMIT:
            # This is a SELL LIMIT order
            if current_price >= self.limit:
                limit_reached = True

        return stop_reached, limit_reached, sl_stop_reached

    def handle_split(self, ratio: float) -> None:
        """
        handle_transaction the amount, limit_price and stop_price
        思考这样的做法是不是后复权呢
        """
        self.amount = int(self.amount / ratio)

        if self.limit is not None:
            self.limit = round(self.limit * ratio, 2)

        if self.stop is not None:
            self.stop = round(self.stop * ratio, 2)

    def set_status(self, status: ORDER_STATUS) -> None:
        self._status = status

    def cancel(self) -> None:
        self._status = ORDER_STATUS.CANCELLED

    def reject(self, reason="") -> None:
        self._status = ORDER_STATUS.REJECTED
        self.reason = reason

    def hold(self, reason="") -> None:
        self._status = ORDER_STATUS.HELD
        self.reason = reason

    @property
    def status(self):
        if not self.open_amount:
            return ORDER_STATUS.FILLED
        elif self._status == ORDER_STATUS.HELD and self.filled:
            return ORDER_STATUS.OPEN
        else:
            return self._status

    @property
    def open(self):
        return self.status in [ORDER_STATUS.OPEN, ORDER_STATUS.HELD]

    @property
    def triggered(self) -> bool:
        """
        for a market order, True
        for a stop order, True iff stop_reached
        for a limit order, True iff limit_reached
        """
        if self.stop is not None and not self.stop_reached:
            return False

        if self.limit is not None and not self.limit_reached:
            return False

        return True

    @property
    def open_amount(self) -> int:
        return self.amount - self.filled

    def __repr__(self):
#         template = """
# {class_name}(
#     id={id},
#     dt={dt},
#     asset={asset},
#     amount={amount},
#     stop={stop},
#     limit={limit},
#     filled={filled},
#     status={status},
# )
#         """
        return f"Order({self.to_dict().__repr__()})"

