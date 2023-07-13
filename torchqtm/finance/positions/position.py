from torchqtm.assets import Asset
from torchqtm.finance.transaction import Transaction
import pandas as pd
import math

import logging

log = logging.getLogger("Performance")


class Position:
    def __init__(
            self,
            asset: Asset,
            amount: int = 0,
            cost_basis: float = .0,
            last_sale_price: float = .0,
            last_sale_date: pd.Timestamp = None
    ):
        self.asset = asset
        self.amount = amount
        self.cost_basis = cost_basis
        self.last_sale_price = last_sale_price
        self.last_sale_date = last_sale_date

    def earn_dividend(self, dividend):
        """
        Register the number of shares we held at this dividend's ex date so
        that we can pay out the correct amount on the dividend's pay date.
        """
        return {"amount": self.amount * dividend.amount}

    # TODO: implement this
    def earn_stock_dividend(self):
        pass

    def handle_split(self, asset: Asset, ratio: float) -> float:
        """
        在分股时会产生小数, 未了在会计上保持一致, 我们return 这部分现金
        """
        assert self.asset == asset

        raw_share = self.amount / ratio

        true_share_count = math.floor(raw_share)

        fractional_share = raw_share - true_share_count

        new_cost_basis = round(self.cost_basis * ratio, 2)

        self.cost_basis = new_cost_basis
        self.amount = true_share_count

        returned_cash = round(float(fractional_share * new_cost_basis), 2)

        log.info("after split: " + str(self))
        log.info("returning cash: " + str(returned_cash))
        return returned_cash

    def handle_transaction(self, txn: Transaction) -> None:
        assert self.asset == txn.asset

        total_amount = self.amount + txn.amount

        if total_amount == 0:
            self.cost_basis = 0.0
        else:
            pre_direction = math.copysign(1, self.amount)
            txn_direction = math.copysign(1, txn.amount)

            if txn_direction != pre_direction:
                if abs(txn.amount) > abs(self.amount):
                    self.cost_basis = txn.price
            else:
                pre_cost = self.cost_basis * self.amount
                txn_cost = txn.price * txn.amount
                total_cost = pre_cost + txn_cost
                self.cost_basis = total_cost / total_amount

            # Update the last sale price if txn is best data we so far
            if self.last_sale_date is None or txn.dt > self.last_sale_date:
                self.last_sale_price = txn.price
                self.last_sale_date = txn.dt

        self.amount = total_amount

    # TODO: implement this
    def handle_commission(self, asset: Asset, cost: float) -> None:
        """

        """
        pass

    def __repr__(self):
        template = """Position(asset: {asset}, amount: {amount}, cost_basis: {cost_basis}, last_sale_price: {last_sale_price})
        """
        return template.format(
            asset=self.asset,
            amount=self.amount,
            cost_basis=self.cost_basis,
            last_sale_price=self.last_sale_price
        )

    def to_dict(self):
        return {
            "sid": self.asset,
            "amount": self.amount,
            "cost_basis": self.cost_basis,
            "last_sale_price": self.last_sale_price
        }




