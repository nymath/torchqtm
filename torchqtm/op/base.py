from abc import ABCMeta, abstractmethod
from torchqtm.vbt.backtest import BackTestEnv


class Parameters(object, metaclass=ABCMeta):
    def __init__(self, data):
        self.data = data


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



