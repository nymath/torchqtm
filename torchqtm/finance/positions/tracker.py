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
# from zipline.utils.sentinel import sentinel
import typing
from torchqtm.data.data_portal import DataPortal
from torchqtm.assets import Future
# from torchqtm.finance._finance_ext import PositionStats
# calculate_positions_tracker_stats
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

        self._need_update_stats = True  # ask chatgpt why use the word "dirty"
        self._stats = PositionStats()  # TODO: implement this

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
        self._need_update_stats = True

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
        self._need_update_stats = True

        asset = txn.asset

        if asset not in self.data:
            position = Position(asset)
            self.data[asset] = position
        else:
            position = self.data[asset]

        position.handle_transaction(txn)

    def handle_commission(self, asset: Asset, cost: float) -> None:
        if asset in self.data:
            self._need_update_stats = True
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
                self._need_update_stats = True

                position = self.data[asset]
                leftover_cash = position.handle_split(asset, ratio)
                total_leftover_cash += leftover_cash

        return total_leftover_cash

    def earn_dividends(self):
        # set _need_stats_update to True
        pass

    def pay_dividends(self):
        # set _need_stats_update to True
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
        # 现在理解作用了吧
        self._need_update_stats = True

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

        Returns
        -------
        stats: PositionStats
            The current stats position stats

        Notes
        -----
        This is cached, repeated access will not recompute the stats until
        the stats may have changed.
        """
        if self._need_update_stats:
            self._stats.calculate_positions_tracker_stats(self.data)
            self._need_update_stats = False
        return self._stats


class PositionStats:
    """Compute values from the current positions

    Attributes
    ----------
    """

    def __init__(self):
        self.gross_exposure = 0
        self.gross_exposure = 0
        self.gross_value = 0
        self.long_exposure = 0
        self.long_value = 0
        self.net_exposure = 0
        self.net_value = 0
        self.short_exposure = 0
        self.short_value = 0
        self.longs_count = 0
        self.shorts_count = 0

    def calculate_positions_tracker_stats(self, positions):
        long_value = 0.0
        short_value = 0.0
        long_exposure = 0.0
        short_exposure = 0.0
        longs_count = 0
        shorts_count = 0

        for position in positions.values():
            # NOTE: this loop does a lot of stuff!
            # we call this function every time the portfolio value is needed,
            # which is at least once per simulation day, so let's not iterate
            # through every single position multiple times.
            exposure = position.amount * position.last_sale_price

            if type(position.asset) is Future:
                # Futures don't have an inherent position value.
                value = 0

                # unchecked cast, this is safe because we do a type check above
                exposure *= position.asset.price_multiplier
            else:
                value = exposure

            if exposure > 0:
                longs_count += 1
                long_value += value
                long_exposure += exposure
            elif exposure < 0:
                shorts_count += 1
                short_value += value
                short_exposure += exposure

            # index[ix] = position.asset.sid
            # position_exposure[ix] = exposure
            # ix += 1

        net_value = long_value + short_value
        gross_value = long_value - short_value

        net_exposure = long_exposure + short_exposure
        gross_exposure = long_exposure - short_exposure

        self.gross_exposure = gross_exposure
        self.gross_value = gross_value
        self.long_exposure = long_exposure
        self.long_value = long_value
        self.longs_count = longs_count
        self.net_exposure = net_exposure
        self.net_value = net_value
        self.short_exposure = short_exposure
        self.short_value = short_value
        self.shorts_count = shorts_count











