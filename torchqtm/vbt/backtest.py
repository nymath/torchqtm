from abc import ABCMeta, abstractmethod
from torchqtm.utils.universe import Universe
import pandas as pd
import numpy as np
from multiprocessing import Pool
from typing import Dict, Hashable
import functools
from torchqtm.base import BackTestEnv
import sklearn.base


class BaseTest(object, metaclass=ABCMeta):
    def __init__(self,
                 env: BackTestEnv = None,
                 universe: Universe = None,
                 *args, **kwargs):
        self.returns = None
        self.env = env
        self._check_env()
        self.universe = universe
        if isinstance(universe, Universe):
            self.symbols = universe.get_symbols()
        self.rebalance_dates = env.dates

    def _check_env(self):
        assert '_FutureReturn' in self.env
        assert 'MktVal' in self.env
        assert 'Sector' in self.env
        assert 'Close' in self.env

    def _reset(self):
        self.returns = None

    @abstractmethod
    def run_backtest(self, modified_factor) -> None:
        raise NotImplementedError("Should implement in the derived class.")


class QuickBackTesting01(BaseTest):
    def __init__(self,
                 env: BackTestEnv,
                 universe: Universe,
                 n_groups: int = 5):
        super().__init__(env, universe)
        self.n_groups = n_groups

    def run_backtest(self, modified_factor) -> None:
        assert modified_factor.shape == self.env['Close'].shape
        self._reset()
        self.env['modified_factor'] = modified_factor
        labels = ["group_" + str(i + 1) for i in range(self.n_groups)]
        returns = []
        for i in range(len(self.env['modified_factor'])-1):
            # If you are confused about concat series, you apply use the following way
            # 1. series.unsqueeze(1) to generate an additional axes
            # 2. concat these series along axis1
            temp_data = pd.concat([self.env._FutureReturn.iloc[i],
                                   self.env.MktVal.iloc[i],
                                   self.env['modified_factor'].iloc[i]], axis=1)
            temp_data.columns = ['_FutureReturn', 'MktVal', 'modified_factor']
            # na stands for stocks that we you not insterested in
            # We can develop a class to better represent this process.
            temp_data = temp_data.loc[~np.isnan(temp_data['modified_factor'])]
            if len(temp_data) == 0:
                group_return = pd.Series(0, index=labels)
            else:
                temp_data['group'] = pd.qcut(temp_data['modified_factor'], self.n_groups, labels=labels)

                def temp(x):
                    # TODO: develop a weight_scheme class
                    weight = x['MktVal'] / x['MktVal'].sum()
                    # weight = 1 / len(x['MktVal'])
                    # weights.append(weight)
                    ret = x['_FutureReturn']
                    return (weight * ret).sum()
                group_return = temp_data.groupby('group').apply(temp)
            returns.append(group_return)
        returns.append(pd.Series(np.repeat(0, self.n_groups), index=group_return.index))
        self.returns = pd.concat(returns, axis=1).T
        # Here we need to transpose the return, since the rows are stocks.
        self.returns.index = self.rebalance_dates
        self.returns.index.name = "trade_date"
        self.returns.columns.name = "group"


class QuickBackTesting02(BaseTest):
    def __init__(self,
                 env: BackTestEnv = None,
                 universe: Universe = None,
                 n_groups: int = 5):
        super().__init__(env, universe)
        self.n_groups = n_groups

    @staticmethod
    def compute_group_return(args):
        i, env, n_groups, labels = args
        temp_data = pd.concat([env['_FutureReturn'].iloc[i],
                               env['MktVal'].iloc[i],
                               env['modified_factor'].iloc[i]], axis=1)
        temp_data.columns = ['_FutureReturn', 'MktVal', 'modified_factor']
        temp_data = temp_data.loc[~np.isnan(temp_data['modified_factor'])]
        temp_data['group'] = pd.qcut(temp_data['modified_factor'], n_groups, labels=labels)

        def temp(x):
            weight = x['MktVal'] / x['MktVal'].sum()
            ret = x['_FutureReturn']
            return (weight * ret).sum()

        return temp_data.groupby('group').apply(temp)

    def run_backtest(self, modified_factor) -> None:
        assert modified_factor.shape == self.env['Close'].shape
        self._reset()
        self.env['modified_factor'] = modified_factor
        labels = ["group_" + str(i + 1) for i in range(self.n_groups)]

        with Pool(4) as pool:
            args = [(i, self.env, self.n_groups, labels) for i in range(len(self.env['modified_factor']))]
            returns = pool.map(self.compute_group_return, args)

        self.returns = pd.concat(returns, axis=1).T
        self.returns.index = self.rebalance_dates
        self.returns.index.name = "trade_date"
        self.returns.columns.name = "group"


# class GPTestingIC(BaseTest):
#     def __init__(self,
#                  env: BackTestEnv,
#                  universe: Universe):
#         super().__init__(env, universe)
#
#     def run_backtest(self, modified_factor) -> float:
#         modified_factor = pd.DataFrame(modified_factor, index=self.env._FutureReturn.index, columns=self.env._FutureReturn.columns)
#         rlt = F.cs_corr(modified_factor, self.env._FutureReturn, 'spearman')
#         if rlt.std() == 0:
#             return -1
#         else:
#             return np.abs(rlt.mean() / rlt.std())
