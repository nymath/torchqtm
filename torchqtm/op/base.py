from abc import ABCMeta, abstractmethod
from torchqtm.vbt.backtest import BackTestEnv


class Parameters(object, metaclass=ABCMeta):
    def __init__(self, data):
        self.data = data


__TYPES = ['momentum', 'reversion']


class Alpha(object, metaclass=ABCMeta):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        self.env = env
        self.args = args
        self.kwargs = kwargs
        self.data = None
        self.type = None

    @abstractmethod
    def operate(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.operate(*args, **kwargs)


class Fundamental(Alpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'fundamental'

    @abstractmethod
    def operate(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError


class Momentum(Alpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'momentum'

    @abstractmethod
    def operate(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError


class Reversion(Alpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'reversion'

    @abstractmethod
    def operate(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError

# # You can use the following format to create your own alpha
# class NeutralizePE(Fundamental):
#     def __init__(self, env: BackTestEnv, *args, **kwargs):
#         super().__init__(env, *args, **kwargs)
#
#     def operate(self, *args, **kwargs):
#         self.rawdata = F.winsorize(self.env.PE, 0.05)
#         self.rawdata = F.normalize(self.rawdata)
#         self.rawdata = F.group_neutralize(self.rawdata, self.env.Sector)
#         self.rawdata = F.regression_neut(self.rawdata, self.env.MktVal)
#         return self.rawdata



