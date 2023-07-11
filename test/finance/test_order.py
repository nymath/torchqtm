from torchqtm.finance.orders.order import Order
from torchqtm.assets import Equity
from torchqtm.types import ORDER_TYPE
import pandas as pd

import unittest
order = Order(
                dt=pd.Timestamp("2005-01-04"),
                asset=Equity("000001.XSHE"),
                amount=10,
                stop=None,
                limit=10)


class TestOrder(unittest.TestCase):

    def test_init(self):
        print(order.id)
        print(order.type)
        self.assertEqual(order.type, ORDER_TYPE.LIMIT)

    def test_get_stop_limit_price(self):
        print(order.stop)
        print(order.limit)

    def test_check_trigger(self):
        self.assertEqual(order.check_order_triggers(8), (False, True, False))
        print(order.check_order_triggers(8))

    def test_property(self):
        self.assertEqual(order.open_amount, 10)
        self.assertEqual(order.open, True)
        print(order.open_amount)
        print(order.open)
        print(order.triggered)


