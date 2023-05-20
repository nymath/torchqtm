from abc import ABCMeta, abstractmethod
from .universe import StaticUniverse
import pandas as pd
import numpy as np


class BackTestEnv(object):
    """
    回测环境, 作为字典的拓展, dfs和env都可以拓展为BackTestEnv
    """
    def __init__(self, dfs, rebalance_dates=None, symbols=None):
        """

        :param dfs:
        :param rebalance_dates:
        :param symbols: 构造的表的columns, 和Universe的概念不同, 我们的因子可以在更大的范围内计算, 但是只在universe上进行回测
        """
        assert isinstance(dfs, dict)
        self.dfs = dfs
        self._check_dfs()
        self.symbols = symbols
        self.rebalance_dates = rebalance_dates
        self.Open = None
        self.High = None
        self.Low = None
        self.Close = None
        self.Volume = None
        self.MktVal = None
        self.PE = None
        self.Sector = None
        self._create_datas()
        self._create_features()

    def _check_dfs(self):
        """
        check is dfs is a valid dict
        :return: None
        """
        assert 'Close' in self.dfs
        assert 'MktVal' in self.dfs
        assert 'Sector' in self.dfs

    def _create_datas(self):
        self.datas = {}
        for key in self.dfs:
            self.datas[key] = self.dfs[key].loc[self.rebalance_dates, self.symbols]
        self.datas['_FutureReturn'] = self.datas['Close'].pct_change().shift(-1)

    def _create_features(self):
        """
        Create the reference to the dict values
        :return:
        """
        for key in self.datas.keys():
            setattr(self, key, self.datas[key])

    def __getitem__(self, item):
        """
        Keep the operator[]
        :param item:
        :return:
        """
        return self.datas[item]

    def __setitem__(self, item, value):
        """
        Keep the operator[]
        :param item:
        :return:
        """
        self.datas[item] = value
        setattr(self, item, value)

    def __delitem__(self, item):
        del self.datas[item]
        delattr(self, item)

    def __contains__(self, item):
        return item in self.datas

    def match_env(self, factor):
        return factor.loc[self.rebalance_dates, self.symbols]


class BackTest(object, metaclass=ABCMeta):
    def __init__(self,
                 env: BackTestEnv = None,
                 universe: StaticUniverse = None,
                 *args, **kwargs):
        self.returns = None
        self.env = env
        self._check_env()
        self.universe = universe
        if isinstance(universe, StaticUniverse):
            self.symbols = universe.get_symbols()
        self.rebalance_dates = env.rebalance_dates

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


class QuickBackTesting01(BackTest):
    def __init__(self,
                 env: BackTestEnv = None,
                 universe: StaticUniverse = None,
                 n_groups: int = 5):
        super().__init__(env, universe)
        self.n_groups = n_groups

    def run_backtest(self, modified_factor) -> None:
        assert modified_factor.shape == self.env['Close'].shape
        self._reset()
        self.env['modified_factor'] = modified_factor
        labels = ["group_" + str(i + 1) for i in range(self.n_groups)]
        returns = []
        for i in range(len(self.env['modified_factor'])):
            temp_data = pd.concat([self.env['_FutureReturn'].iloc[i],
                                   self.env['MktVal'].iloc[i],
                                   self.env['modified_factor'].iloc[i]], axis=1)
            temp_data.columns = ['_FutureReturn', 'MktVal', 'modified_factor']
            # na无法交易, 可以把停牌的股票设置为na
            temp_data = temp_data.loc[~np.isnan(temp_data['modified_factor'])]
            temp_data['group'] = pd.qcut(temp_data['modified_factor'], self.n_groups, labels=labels)

            def temp(x):
                weight = x['MktVal'] / x['MktVal'].sum()
                # weight = 1 / len(x['MktVal'])
                # weights.append(weight)
                ret = x['_FutureReturn']
                return (weight * ret).sum()
            group_return = temp_data.groupby('group').apply(temp)
            returns.append(group_return)
        self.returns = pd.concat(returns, axis=1).T
        self.returns.index = self.rebalance_dates
        self.returns.index.name = "trade_date"
        self.returns.columns.name = "group"

