from abc import ABCMeta, abstractmethod
from torchqtm.utils.universe import Universe
import pandas as pd
import numpy as np
from multiprocessing import Pool
from typing import Dict, Hashable
from torchqtm.base import BackTestEnv
import matplotlib.pyplot as plt
from torchqtm.utils.visualization import ColorGenerator
import torchqtm.op.functional as F
from typing import final


class BaseTester(object, metaclass=ABCMeta):
    def __init__(self,
                 env: BackTestEnv = None):
        self.metrics = None
        self.results = None
        self.env = env
        self._check_env()
        self.rebalance_dates = env.dates

    def _check_env(self):
        ...
        # assert '_FutureReturn' in self.env
        # assert 'MktVal' in self.env
        # assert 'Sector' in self.env
        # assert 'Close' in self.env

    def _reset(self):
        self.metrics = None
        self.results = None

    @abstractmethod
    def run_backtest(self, modified_factor) -> None:
        raise NotImplementedError("Should implement in the derived class.")


class TesterMixin:
    def score(self, alpha, method='pearson'):
        return np.nanmean(F.cs_corr(alpha, self.env.create_forward_returns(), method=method), axis=0)

    # def _more_tags(self):
    #     return {"requires_y": True}


class BaseGroupTester(BaseTester, TesterMixin):
    def __init__(self,
                 env: BackTestEnv = None,
                 n_groups: int = 5,
                 weighting: str = 'equal',
                 exclude_suspended: bool = False,
                 exclude_limits: bool = False):
        """

        Parameters
        ----------
        env
        n_groups
        weighting: "equal" or "market_cap" or "factor_cap"
        exclude_suspended
        exclude_limits
        """
        super().__init__(env)
        self.n_groups = n_groups
        self.weighting = weighting
        self.exclude_limits = exclude_limits
        self.exclude_suspended = exclude_suspended
        self.returns = None

    def _reset(self):
        super()._reset()
        self.returns = None

    @abstractmethod
    def run_backtest(self, modified_factor) -> None:
        raise NotImplementedError("Should implement in the derived class.")

    # TODO: add more parameters
    def plot(self):
        # plot the result
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        color_generator = ColorGenerator(self.n_groups)
        colors = color_generator.data
        for i in range(self.n_groups):
            ax.plot((1 + self.returns.iloc[:, i]).cumprod(), label=f'group_{i + 1}', color=colors[i])
        fig.legend(fontsize=16)
        fig.show()


class BaseIcTester(BaseTester, TesterMixin):
    def __init__(self,
                 env: BackTestEnv = None,
                 universe: Universe = None,
                 method: str = "pearson"):
        super().__init__(env, universe)
        self.method = method
        self.results = None

    def _reset(self):
        super()._reset()

    @abstractmethod
    def run_backtest(self, modified_factor) -> None:
        raise NotImplementedError
        # self._reset()
        # self.results = F.cs_corr(modified_factor, self.env._FutureReturn)
        # self.metrics = np.nanmean(self.results, axis=0)

    @final
    def plot(self):
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(self.rebalance_dates, self.results)
        fig.legend(fontsize=16)
        fig.show()


class GroupTester01(BaseGroupTester):
    def __init__(self,
                 env: BackTestEnv = None,
                 n_groups: int = 5,
                 weighting: str = 'equal',
                 exclude_suspended: bool = False,
                 exclude_limits: bool = False):
        super().__init__(env, n_groups, weighting, exclude_suspended, exclude_limits)

    def run_backtest(self,
                     modified_factor,
                     purify=True) -> None:
        assert modified_factor.shape == self.env['Close'].shape
        if purify:
            modified_factor = F.purify(modified_factor)
        self._reset()
        labels = ["group_" + str(i + 1) for i in range(self.n_groups)]
        returns = []
        forward_return = self.env.create_forward_returns(D=1)
        symbols = list(self.env.Close.columns)

        datas = np.concatenate([forward_return.values[..., np.newaxis],
                                self.env.MktVal.values[..., np.newaxis],
                                modified_factor.values[..., np.newaxis]], axis=2)

        for i in range(len(modified_factor)):

            data_slice = pd.DataFrame(datas[i],
                                      index=symbols,
                                      columns=['forward_returns', 'MktVal', 'modified_factor'])
            data_masked = data_slice.loc[~np.isnan(data_slice['modified_factor'])].copy()
            if len(data_masked) == 0 or data_masked['forward_returns'].isna().all():
                group_return = pd.Series(0, index=labels)
            else:
                data_masked.loc[:, 'group'] = pd.qcut(data_masked['modified_factor'], self.n_groups, labels=labels)

                def temp(x):
                    # TODO: develop a weight_scheme class
                    if self.weighting == 'equal':
                        weight = 1 / len(x['MktVal'])
                    elif self.weighting == 'market_cap':
                        weight = x['MktVal'] / x['MktVal'].sum()
                    elif self.weighting == 'factor_cap':
                        weight = x['modified_factor'] / np.sum(np.abs(x['modified_factor']))
                    else:
                        raise ValueError('Invalid weight scheme')
                    ret = x['forward_returns']
                    return (weight * ret).sum()

                group_return = data_masked.groupby('group').apply(temp)
            returns.append(group_return)
        self.returns = pd.concat(returns, axis=1).T
        # Here we need to transpose the return, since the rows are stocks.
        self.returns.index = self.rebalance_dates
        self.returns.index.name = "trade_date"
        self.returns.columns.name = "group"


class LongShort01(BaseGroupTester):
    def __init__(self,
                 env: BackTestEnv = None,
                 n_groups: int = 5,
                 leverage: float = 1,
                 weighting: str = 'equal',
                 exclude_suspended: bool = False,
                 exclude_limits: bool = False):
        super().__init__(env, n_groups, weighting, exclude_suspended, exclude_limits)
        self.leverage = leverage

    def run_backtest(self,
                     modified_factor,
                     purify=True) -> None:
        """
        self.returns
        """
        assert modified_factor.shape == self.env['Close'].shape
        if purify:
            modified_factor = F.purify(modified_factor)
        self._reset()
        labels = ["group_" + str(i + 1) for i in range(self.n_groups)]
        returns = []
        forward_return = self.env.create_forward_returns(D=1)
        symbols = list(self.env.Close.columns)

        datas = np.concatenate([forward_return.values[..., np.newaxis],
                                self.env.MktVal.values[..., np.newaxis],
                                modified_factor.values[..., np.newaxis]], axis=2)

        for i in range(len(modified_factor)):

            data_slice = pd.DataFrame(datas[i],
                                      index=symbols,
                                      columns=['forward_returns', 'MktVal', 'modified_factor'])
            data_masked = data_slice.loc[~np.isnan(data_slice['modified_factor'])].copy()
            if len(data_masked) == 0 or data_masked['forward_returns'].isna().all():
                group_return = pd.Series(0, index=labels)
            else:
                data_masked.loc[:, 'group'] = pd.qcut(data_masked['modified_factor'], self.n_groups, labels=labels)

                def aggHelper(x):
                    # TODO: develop a weight_scheme class
                    if self.weighting == 'equal':
                        weight = 1 / len(x['MktVal'])
                    elif self.weighting == 'market_cap':
                        weight = x['MktVal'] / x['MktVal'].sum()
                    elif self.weighting == 'factor_cap':
                        weight = x['modified_factor'] / np.sum(np.abs(x['modified_factor']))
                    else:
                        raise ValueError('Invalid weight scheme')
                    ret = x['forward_returns']
                    return (weight * ret).sum()

                group_return = data_masked.groupby('group').apply(aggHelper)
            returns.append(group_return)
        temp_data = pd.concat(returns, axis=1).T
        returns = temp_data[f'group_{self.n_groups+1}'] - temp_data[f'group_{1}']
        if self.leverage == 1:
            self.returns = returns / 2
        else:
            self.returns = returns
        # Here we need to transpose the return, since the rows are stocks.
        self.returns.index = self.rebalance_dates

    @final
    def plot(self):
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot((1 + self.returns).cumprod())
        fig.legend(fontsize=16)
        fig.show()


class WQTester(BaseTester, TesterMixin):

    def __init__(self,
                 env: BackTestEnv = None,
                 exclude_suspended: bool = False,
                 exclude_limits: bool = False
                 ):
        super().__init__(env)
        self.env = env
        self.exclude_suspended = exclude_suspended
        self.returns = None
        self.exclude_limits = exclude_limits

    def _reset(self):
        super()._reset()
        self.returns = None

    def run_backtest(self,
                     modified_factor,
                     purify=True):

        assert modified_factor.shape == self.env['Close'].shape
        if purify:
            modified_factor = F.purify(modified_factor)
        self._reset()
        forward_return = self.env.create_forward_returns(D=1)
        symbols = list(self.env.Close.columns)
        returns = []
        datas = np.concatenate([forward_return.values[..., np.newaxis],
                                self.env.MktVal.values[..., np.newaxis],
                                modified_factor.values[..., np.newaxis]], axis=2)

        for i in range(len(modified_factor)):

            # Step1: Evaluate the expression for each stock to generate the alpha vector for the given date.
            data_slice = pd.DataFrame(datas[i],
                                      index=symbols,
                                      columns=['forward_returns', 'MktVal', 'modified_factor'])
            data_masked = data_slice.loc[~np.isnan(data_slice['modified_factor'])].copy()

            # Step2: From each value in the vector, subtract the average of the vector values in the group.
            # Sum of all vector values = 0. This is called neutralization.
            # The group can be the entire market,
            # but we can also perform this neutralization operation on sector,
            # industry or subindustry groupings of stocks.
            data_masked['modified_factor'] -= np.mean(data_masked['modified_factor'])

            # Step 3: The resulting values are scaled or ‘normalized’ such that absolute sum of the alpha vector
            # values is 1. These values can be called as normalized weights.
            data_masked['modified_factor'] /= np.sum(np.abc(data_masked['modified_factor']))

            # Step 4: Using normalized weights, the BRAIN simulator allocates capital
            # (from a fictitious book of $20 million) to each stock to construct a portfolio.
            returns.append(np.sum(data_masked['modified_factor'] * data_masked['forward_returns']))

        self.returns = pd.Series(returns, index=self.rebalance_dates)
