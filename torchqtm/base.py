from abc import ABCMeta, abstractmethod
from torchqtm.utils.universe import Universe
import pandas as pd
import numpy as np
from typing import Dict, Hashable
from abc import ABCMeta, abstractmethod
from typing import Iterable, Callable
from collections import OrderedDict
from torchqtm.utils._decorators import ContextManager
from torchqtm.utils.warnings import catch_warnings


class BackTestEnv(object):
    """
    # We can create two different BackTestEnvs
    # 1. For Op, we create env with trade_dates
    # 2. For backtest, we create env with rebalance_dates
    """

    def __init__(self,
                 dfs: Dict[Hashable, pd.DataFrame],
                 dates,
                 symbols):
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
        self.shape = (len(self.symbols), len(self.dates))
        self.Open = None
        self.High = None
        self.Low = None
        self.Close = None
        self.Volume = None
        self.Returns = None
        self.Vwap = None
        self.MktVal = None
        self.PE = None
        self.Sector = None
        self.forward_returns = None
        self._create_datas()
        self._create_features()

    def _check_dfs(self):
        """
        check is dfs is a valid dict
        :return: None
        """
        # assert 'close' in self.dfs
        assert 'MktVal' in self.dfs
        assert 'Sector' in self.dfs

    def _create_datas(self):
        self.data = {}
        for key in self.dfs:
            if isinstance(self.dfs[key], pd.DataFrame):
                self.data[key] = self.dfs[key].loc[self.dates, self.symbols]
            else:
                self.data[key] = self.dfs[key]

    def create_forward_returns(self, D=1):
        forward_returns = self.data['Close'].pct_change().shift(-1)
        return forward_returns

    def _create_features(self):
        """
        Create the reference to the dict values
        :return:
        """
        for key in self.data.keys():
            setattr(self, key, self.data[key])

    def __getitem__(self, item):
        """
        Keep the operator[]
        :param item:
        :return:
        """
        return self.data[item]

    def __setitem__(self, item, value):
        """
        Keep the operator[]
        :param item:
        :return:
        """
        assert isinstance(value, pd.DataFrame)
        self.data[item] = value
        setattr(self, item, value)

    def __delitem__(self, item):
        del self.data[item]
        delattr(self, item)

    def __contains__(self, item):
        return item in self.data

    def match(self, factor):
        return factor.loc[self.dates, self.symbols]


class Parameter(np.ndarray, metaclass=ABCMeta):
    def __new__(cls,
                data: int,
                requires_optim: bool = False,
                feasible_region: Iterable[int] = None):
        obj = np.asarray(data).view(cls)
        obj.required_optim = requires_optim
        obj.feasible_region = feasible_region
        obj._check_data()
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.required_optim = getattr(obj, 'required_optim', None)
        self.feasible_region = getattr(obj, 'feasible_region', None)

    def _check_data(self):
        if self.required_optim is True and self.feasible_region is None:
            raise ValueError("must provide possible values for search")


__TYPES = ['momentum', 'reversion']


class BaseOperator(object, metaclass=ABCMeta):

    catch_warnings = False

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BaseAlpha(BaseOperator, metaclass=ABCMeta):

    __DATA_AVAILABLE__ = ['open', 'high', 'low', 'close', 'volume',
                          'returns', 'vwap', 'adv20', 'sector']

    def __init__(self, env: BackTestEnv, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.env = env
        self.args = args
        self.kwargs = kwargs
        self.data = None
        self.type = None
        self.Enabled = True  # default to be used

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError

    def __eq__(self, other):
        return self.__repr__() == other.__repr__() and self.data == other.data

    @property
    def open(self):
        return self.env.Open

    @property
    def high(self):
        return self.env.High

    @property
    def low(self):
        return self.env.Low

    @property
    def close(self):
        return self.env.Close

    @property
    def volume(self):
        return self.env.Volume

    @property
    def returns(self):
        return self.env.Returns

    @property
    def vwap(self):
        return self.env.Vwap

    @property
    def adv20(self):
        return self.env.adv20

    @property
    def sector(self):
        return self.env.Sector


class Volatility(BaseAlpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'volatility'

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError


class Fundamental(BaseAlpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'fundamental'

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError


class Momentum(BaseAlpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'momentum'

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError


class Reversion(BaseAlpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'reversion'

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError


class Technical(BaseAlpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'technical'

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError




