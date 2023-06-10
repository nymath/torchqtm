from abc import ABCMeta, abstractmethod
from torchqtm.vbt.backtest import BackTestEnv
import numpy as np
from typing import Iterable


import numpy as np
from abc import ABCMeta
from typing import Iterable


class Parameter(np.ndarray, metaclass=ABCMeta):
    def __new__(cls, data: int, required_optim: bool = False, feasible_region: Iterable[int] = None):
        obj = np.asarray(data).view(cls)
        obj.required_optim = required_optim
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
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BaseAlpha(BaseOperator, metaclass=ABCMeta):
    def __init__(self, env: BackTestEnv, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.env = env
        self.args = args
        self.kwargs = kwargs
        self.data = None
        self.type = None

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError

    def __eq__(self, other):
        return self.__repr__() == other.__repr__() and self.data == other.data

    @property
    def open(self):
        return self.env.Close

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

# # You can use the following format to create your own alpha
# class NeutralizePE(Fundamental):
#     def __init__(self, env: BackTestEnv, *args, **kwargs):
#         super().__init__(env, *args, **kwargs)
#
#     def forward(self, *args, **kwargs):
#         self.rawdata = F.winsorize(self.env.PE, 0.05)
#         self.rawdata = F.normalize(self.rawdata)
#         self.rawdata = F.group_neutralize(self.rawdata, self.env.Sector)
#         self.rawdata = F.regression_neut(self.rawdata, self.env.MktVal)
#         return self.rawdata



