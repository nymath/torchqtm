import numpy as np
import pandas as pd
import torchqtm.op.functional as F
from typing import Optional
import matplotlib.pyplot as plt


class RiskEvaluator(object):
    def __init__(self,
                 returns: pd.Series,
                 weights: pd.DataFrame = None,
                 freq=None,
                 rfr=0,
                 benchmark=None):
        self.returns = returns
        self.weights = weights
        self.freq = freq
        self.rfr = rfr
        self.benchmark = benchmark
        self.perf = {}

    def _sharpe(self):
        return sharpe(self.returns, self.freq, self.rfr)

    def _max_draw_down(self):
        series, val, longest_time = drawdown(self.returns)
        return series, val, longest_time

    def _turn_over(self):
        assert self.weights is not None
        return turnover(self.weights)

    def _beta(self):
        assert self.benchmark is not None
        pass

    def _alpha(self):
        assert self.benchmark is not None
        pass

    def _monthly_returns(self):
        pass

    def _yearly_returns(self):
        pass

    def _annualized_mean(self):
        pass

    def _annualized_volatility(self):
        pass


class FactorEvaluator(object):
    def __init__(self,
                 factor_scores: pd.DataFrame,
                 forward_returns: pd.DataFrame):
        self.factor_scores = factor_scores
        self.forward_returns = forward_returns

    def _ic(self):
        return ic(self.factor_scores, self.forward_returns)


def ic(factor_scores: pd.DataFrame, forward_returns: pd.DataFrame, method='pearson') -> pd.Series:
    """
    :return: id series
    """
    assert factor_scores.shape == forward_returns.shape
    rlt = F.cs_corr(factor_scores, forward_returns, method=method)
    rlt.fillna(0, inplace=True)
    return rlt


def drawdown(returns: pd.Series) -> (pd.Series, float, int):
    """
    :returns : drawdown series, max drawdown, longest drawdown
    """
    net_curve = (1+returns).cumpord()
    cmax = net_curve.cumax()
    rlt = cmax / net_curve - 1
    longest_drawdown = cmax.value_counts().max() - 1
    if isinstance(returns, pd.Series):
        rlt = pd.Series(rlt, index=returns.index)
    elif isinstance(returns, np.ndarray):
        rlt = np.array(rlt)
    return rlt, rlt.max(), longest_drawdown


def turnover(weights: pd.DataFrame) -> pd.Series:
    """
    calculate the amount traded at each rebalance rawdata as fraction of portfolio size.
    """
    weights.fillna(0, inplace=True)
    forward_weights = weights.shift(1)
    forward_weights.fillna(0, inplace=True)
    rlt = (weights - forward_weights).abs().sum(axis=1)
    return rlt


def sharpe(returns, freq=None, rfr=0):
    """
    Calculates the Sharpe ratio from a returns time series and the risk-free rate.

    Parameters
    ----------
    period_returns : DataSeries
        pandas series indexed by date with returns for each period as fraction of previous value
        (i.e. not in percent)
    freq : {'A', 'Q', 'M', 'D', or None}, optional, default None
        The frequency of the rawdata. If not None will annualize the result by multiplying by
        :math:`\sqrt{N}`
        where :math:`N` is the number of periods per year for the respective frequency
        (e.g. 252 for daily, 12 for monthly, 60 for weekly, etc.)
    rfr : float or pandas time series, default 0.
        Risk free rate, scaled to match returns rawdata frequency (i.e. annual if annual returns
        are provided, etc.)

    """
    if returns.isna().sum() > 0:
        raise ValueError('period_returns contains nan values.')

    mean = returns.mean()
    std = returns.std()
    _annf = freq
    return mean / std * _annf ** 0.5







