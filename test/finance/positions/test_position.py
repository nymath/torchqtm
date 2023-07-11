from torchqtm.finance.positions.position import Position
from torchqtm.assets import Equity
from torchqtm.finance.orders.order import Order
from torchqtm.finance.transaction import Transaction
from torchqtm.assets import Equity
from torchqtm.types import ORDER_TYPE
import pandas as pd
import unittest

dt = pd.Timestamp("2005-01-04")
asset = Equity("000001.XSHE")
order = Order(dt=dt,
              asset=asset,
              amount=100,
              stop=None,
              limit=10.)

transaction = Transaction(asset=asset, amount=100, dt=dt, price=10., order_id=order.id)

class TestPosition(unittest.TestCase):
    def test_init(self):
        position = Position(asset=asset, amount=0)
        self.assertEqual(position.asset, asset)
        self.assertEqual(position.amount, 0)

    def test_update_from_transaction(self):
        position = Position(asset=asset, amount=0)
        position.handle_transaction(transaction)
        self.assertEqual(position.amount, 100)
        self.assertEqual(position.cost_basis, 10)
        print(position.cost_basis)
        print(position.last_sale_date)
        print(position.last_sale_price)
        print(position.amount)

    def test_handle_split(self):
        position = Position(asset=asset, amount=0)
        position.handle_transaction(transaction)
        position.handle_split(asset=asset, ratio=0.5)
        print(position.amount)
        print(position.cost_basis)







