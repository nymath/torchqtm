# TODO: implement it

from collections import OrderedDict
from functools import partial
import math
import copy
import logging
import numpy as np
import pandas as pd
from torchqtm.assets import Asset
from torchqtm.finance.transaction import Transaction
from torchqtm.finance.positions.position import Position
from zipline.utils.sentinel import sentinel
import typing
from torchqtm.data.data_portal import DataPortal

log = logging.getLogger("PositionsTracker")


class PositionsTracker(object):
    def __init__(self, data_frequency: typing.Literal["daily", "minute"]):
        if data_frequency not in {"daily", "minute"}:
            raise ValueError("data_frequency must be one of 'daily', 'minute'")
        self.data_frequency = data_frequency
        self.data: typing.OrderedDict[Asset, Position] = OrderedDict()
        self._unpaid_dividends = {}
        self._unpaid_stock_dividends = {}
        self._positions_store = {}

        self._dirty_stats = True  # ask chatgpt why use the word "dirty"
        self._stats = None  # TODO: implement this

    def __getitem__(self, item: Asset):
        return self.data[item]

    def update_position(
            self,
            asset: Asset,
            amount: int = None,
            last_sale_price: float = None,
            last_sale_date: pd.Timestamp = None,
            cost_basis: float = None,
    ):
        self._dirty_stats = True

        if asset not in self.data:
            position = Position(asset)
            self.data[asset] = position

        else:
            position = self.data[asset]

        if amount is not None:
            position.amount = amount
        if last_sale_price is not None:
            position.last_sale_price = last_sale_price
        if last_sale_date is not None:
            position.last_sale_date = last_sale_date
        if cost_basis is not None:
            position.cost_basis = cost_basis

    def handle_transaction(self, txn: Transaction):
        self._dirty_stats = True

        asset = txn.asset

        if asset not in self.data:
            position = Position(asset)
            self.data[asset] = position
        else:
            position = self.data[asset]

        position.handle_transaction(txn)

    def handle_commission(self, asset: Asset, cost: float) -> None:
        if asset in self.data:
            self._dirty_stats = True
            self.data[asset].handle_commission(asset, cost)

    def handle_splits(self, splits: typing.List[typing.Tuple[Asset, float]]) -> float:
        """Process a list of splits by modifying any positions as needed
        Returns
        -------
        int: The leftover cash from fractional shares after modifying each position
        """
        total_leftover_cash = 0

        for asset, ratio in splits:
            if asset in self.data:
                self._dirty_stats = True
                position = self.data[asset]
                leftover_cash = position.handle_split(asset, ratio)
                total_leftover_cash += leftover_cash

        return total_leftover_cash

    def earn_dividends(self):
        pass

    def pay_dividends(self):
        pass

    def maybe_create_position_transaction(self):
        pass

    def get_positions(self) -> typing.Dict[Asset, Position]:
        # positions = self._positions_store
        #
        # for asset, position in self.positions.items():
        #     positions[asset] = position
        # TODO: 尽量只写
        return self.data

    def get_position_list(self) -> typing.List[Position]:
        return [
            pos for asset, pos in self.data.items() if pos.amount != 0
        ]

    def sync_last_sale_price(
            self,
            dt: pd.Timestamp,
            data_portal: DataPortal,
            handle_non_market_minutes=False,
    ):
        self._dirty_stats = True

        if handle_non_market_minutes:
            raise NotImplementedError

        for position in self.data.values():
            position.last_sale_price = data_portal.get_scalar_asset_spot_value(
                asset=position.asset,
                filed="price",
                dt=dt,
                data_frequency=self.data_frequency,
            )
            position.last_sale_date = dt

    @property
    def stats(self):
        """The current status of the positions
        """
        if self._dirty_stats:
            calculate_position_tracker_stats(self.data, self._stats)
            self._dirty_stats = False
        return self._stats













