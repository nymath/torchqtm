from typing import Tuple
import unittest

import pandas as pd

from torchqtm.finance.models.slippage import SlippageModel
from torchqtm.data.data_portal import DataPortal
from torchqtm.finance.orders.order import Order
from torchqtm.assets import Equity


class TestModel(SlippageModel):
    def process_order(self, data: DataPortal, order: Order) -> Tuple[float, int]:
        return 10.0, 5


model = TestModel()
simulation_dt_func = lambda: pd.Timestamp("20100304")

data_portal = DataPortal(simulation_dt_func=simulation_dt_func)
asset = Equity("000001.XSHE")


order = Order(pd.Timestamp("20100304"),
              Equity("000001.XSHE"),
              100)


class test_slippage(object):
    def test_process_order(self):
        rlt = data_portal.get_scalar_asset_spot_value(Equity("000001.XSHE"), "close", pd.Timestamp("20100304"), "daily")
        aa = model.simulate(data_portal, asset, [order])
        print(rlt)


test_slippage().test_process_order()