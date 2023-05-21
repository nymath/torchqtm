from abc import ABCMeta, abstractmethod
from quant.bt.universe import StaticUniverse
import pandas as pd
import numpy as np
from .dataset import QuantDataFrame
from multiprocessing import Pool, cpu_count


# TODO: convert all the data type to the QuantDataFrame
class BackTestEnv(object):
    """
    回测环境, 作为字典的拓展, dfs和env都可以拓展为BackTestEnv
    # We can create two different BackTestEnvs
    # 1. For Op, we create env with trade_dates
    # 2. For backtest, we create env with rebalance_dates
    """

    def __init__(self, dfs, dates=None, symbols=None):
        """

        :param dfs:
        :param dates:
        :param symbols: 构造的表的columns, 和Universe的概念不同, 我们的因子可以在更大的范围内计算, 但是只在universe上进行回测
        """
        assert isinstance(dfs, dict)
        self.dfs = dfs
        self._check_dfs()
        self.symbols = symbols
        self.dates = dates
        self.Open = None
        self.High = None
        self.Low = None
        self.Close = None
        self.Volume = None
        self.MktVal = None
        self.PE = None
        self.Sector = None
        self._FutureReturn = None
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
            if isinstance(self.dfs[key], pd.DataFrame):
                self.datas[key] = self.dfs[key].loc[self.dates, self.symbols]
            else:
                self.datas[key] = self.dfs[key]
        self.datas['_FutureReturn'] = self.datas['Close'].pct_change().shift(-1)

    def _create_features(self):
        """
        Create the reference to the dict values
        :return:
        """
        for key in self.datas.keys():
            setattr(self, key, self.datas[key])
        setattr(self, '_FutureReturn', self.datas['_FutureReturn'])

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
        return factor.loc[self.dates, self.symbols]


class BaseTest(object, metaclass=ABCMeta):
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
        self.returns = pd.concat(returns, axis=1).T
        # Here we need to transpose the return, since the rows are stocks.
        self.returns.index = self.rebalance_dates
        self.returns.index.name = "trade_date"
        self.returns.columns.name = "group"


class QuickBackTesting02(BaseTest):
    def __init__(self,
                 env: BackTestEnv = None,
                 universe: StaticUniverse = None,
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
