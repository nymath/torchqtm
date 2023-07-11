from copy import copy
import typing
import pandas as pd
from torchqtm.finance.orders.order import Order
from torchqtm.assets import Asset


class Transaction:
    def __init__(
            self,
            asset: Asset,
            amount: int,
            dt: pd.Timestamp,
            price: float,
            order_id: str,
    ):
        self.asset = asset
        self.amount = amount
        self.dt = dt
        self.price = price
        self.order_id = order_id
        # TODO: implement type
        self.type = None

    def __getitem__(self, item):
        return self.__dict__[item]

    def __repr__(self):
        template = "{cls}(asset={asset}, dt={dt}," " amount={amount}, price={price})"

        return template.format(
            cls=type(self).__name__,
            asset=self.asset,
            dt=self.dt,
            amount=self.amount,
            price=self.price,
        )

    def to_dict(self):
        # TODO: implement this
        pass


def create_transaction(order, amount: int, dt: pd.Timestamp, price: float):
    """Create a transaction through a given order
    我感觉没必要用这个函数
    """
    transaction = Transaction(
        asset=order.asset,
        amount=amount,
        dt=dt,
        price=price,
        order_id=order.id
    )
    return transaction
