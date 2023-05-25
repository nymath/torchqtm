from abc import ABCMeta, abstractmethod
from quant.vbt.backtest import BackTestEnv


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
        """assign self.data and return"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.operate(*args, **kwargs)


class Fundamental(Alpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'fundamental'

    @abstractmethod
    def operate(self, *args, **kwargs):
        """assign self.data and return"""
        raise NotImplementedError


class Momentum(Alpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'momentum'

    @abstractmethod
    def operate(self, *args, **kwargs):
        """assign self.data and return"""
        raise NotImplementedError


class Reversion(Alpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'reversion'

    @abstractmethod
    def operate(self, *args, **kwargs):
        """assign self.data and return"""
        raise NotImplementedError

# # You can use the following format to create your own alpha
# class NeutralizePE(Fundamental):
#     def __init__(self, env: BackTestEnv, *args, **kwargs):
#         super().__init__(env, *args, **kwargs)
#
#     def operate(self, *args, **kwargs):
#         self.data = F.winsorize(self.env.PE, 0.05)
#         self.data = F.normalize(self.data)
#         self.data = F.group_neutralize(self.data, self.env.Sector)
#         self.data = F.regression_neut(self.data, self.env.MktVal)
#         return self.data



