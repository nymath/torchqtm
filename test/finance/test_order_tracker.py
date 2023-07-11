from torchqtm.finance.orders.order import Order
from torchqtm.finance.orders.tracker import OrdersTracker
from torchqtm.assets import Equity
from torchqtm.types import ORDER_TYPE
import pandas as pd
from torchqtm.data.data_portal import DataPortal
import unittest

order = Order(
                dt=pd.Timestamp("2005-01-04"),
                asset=Equity("000001.XSHE"),
                amount=10,
                stop=None,
                limit=10)
symbol = Equity("000001.XSHE")

simulation_dt_func = lambda: pd.Timestamp("2005-01-04")

data_portal = DataPortal(simulation_dt_func=simulation_dt_func)

tracker = OrdersTracker()
tracker.order(asset=symbol, amount=100)

tracker.get_transactions(data_portal)


class test_orders_tracker(unittest.TestCase):
    def test_init(self):
        print("--------------------------------")
        print(tracker.slippage_models)
        print(tracker.commission_models)

    def test_get_transactions(self):
        tracker.order(asset=symbol, amount=100)
        print(tracker.get_transactions(data_portal))
