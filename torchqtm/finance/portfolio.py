import pandas as pd
import numpy as np
from torchqtm.finance.positions.tracker import PositionsTracker
import typing


class Portfolio:
    """object providing information about current portfolio state

    Parameters
    ----------
    start_date : pd.Timestamp
        The start date for the period being recorded.
    capital_base : float
        The starting value for the portfolio. This will be used as the starting
        cash, current cash, and portfolio value.

    Attributes
    ----------
    positions : zipline.protocol.Positions
        Dict-like object containing information about currently-held positions.
    cash : float
        Amount of cash currently held in portfolio.
    portfolio_value : float
        Current liquidation value of the portfolio's holdings.
        This is equal to ``cash + sum(shares * price)``
    starting_cash : float
        Amount of cash in the portfolio at the start of the backtest.
    """
    def __init__(
            self,
            start_date: pd.Timestamp,
            capital_base: float = 0.0,
    ):
        self.cash_flow: float = 0.0
        self.starting_cash: float = capital_base
        self.portfolio_value: float = capital_base
        self.pnl: float = 0.0
        self.returns: float = 0.0
        self.cash: float = 0.0
        self.positions: typing.Dict = {}
        self.start_date: pd.Timestamp = start_date
        self.positions_value: float = 0.0
        self.positions_exposure: float = 0.0

    @property
    def capital_used(self):
        return self.cash_flow

    def __setattr__(self, key, value):
        raise AttributeError

    def __repr__(self):
        return "Portfolio({0})".format(self.__dict__)

    @property
    def current_portfolio_weights(self):
        """
        Compute each asset's weight in the portfolio by calculating its held
        value divided by the total value of all positions

        Each equity's value is its price times the number of shares held. Each
        futures contract's value is its unit price times number of shares held
        times the multiplier.
        """

        position_values = pd.Series(
            {
                asset: (
                        position.last_sale_price * position.amount * asset.price_multiplier
                )
                for asset, position in self.positions.items()
            },
            dtype=np.float64,
        )
        return position_values / self.portfolio_value








