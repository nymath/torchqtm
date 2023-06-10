import torchqtm.op as op
import torchqtm.op.functional as F
from torchqtm.op.functional import *
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from torchqtm.vbt.backtest import BackTestEnv
from torchqtm.op.base import BaseAlpha

__all__ = [f"Alpha{str(i).zfill(3)}" for i in range(1, 102)]


class WQAlpha101(BaseAlpha):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'WorldQuant Alpha101'

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError


class Alpha001(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = -1 * ts_rank(cs_rank(self.low), 9)
        return self.data


class Alpha002(WQAlpha101):
    """
     (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = -1 * ts_corr(cs_rank(ts_delta(log(self.volume), 2)), cs_rank((self.close - self.open) / self.open), 6)
        return self.data


class Alpha003(WQAlpha101):
    """
    The Spearman correlation of open and volume
    (-1 * correlation(rank(open), rank(volume), 10))
    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = -1 * ts_corr(cs_rank(self.open), cs_rank(self.volume), 10)
        return self.data


class Alpha004(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = -1 * ts_rank(cs_rank(self.low), 9)
        return self.data


class Alpha005(WQAlpha101):
    """
     (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = (cs_rank((self.open - ts_mean(self.vwap, 10))) * (-1 * abs(cs_rank((self.close - self.vwap)))))
        return self.data


class Alpha006(WQAlpha101):
    """
     (-1 * correlation(open, volume, 10))
    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = -1 * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha007(WQAlpha101):
    """
    ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        condition = self.adv20 < self.volume
        value_if_true = -1 * ts_rank(abs(ts_delta(self.close, 7)), 60) * sign(ts_delta(self.close, 7))
        value_if_false = -1
        self.data = if_else(condition, value_if_true, value_if_false)
        return self.data


class Alpha008(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = -1 * (cs_rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                                   ts_delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))
        return self.data

    def __repr__(self):
        return "Alpha: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))"


class Alpha009(WQAlpha101):
    """
     (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = -1 * ts_corr(cs_rank(ts_delta(log(self.volume), 2)), cs_rank((self.close - self.open) / self.open), 6)
        return self.data

    def __repr__(self):
        return "(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"


class Alpha011(WQAlpha101):
    """
    ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = ((F.cs_rank(F.ts_max((self.env.Vwap - self.env.Close), 3)) +
                      F.cs_rank(F.ts_min((self.env.Vwap - self.env.Close), 3))) * F.cs_rank(
            F.ts_delta(self.env.Volume, 3)))
        return self.data

    def __repr__(self):
        return "((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))"


class Alpha012(WQAlpha101):
    """
    (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = sign(ts_delta(self.volume, 1)) * (-1 * ts_delta(self.close, 1))
        return self.data

    def __repr__(self):
        return "(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"


# TODO: implement it
class Alpha013(WQAlpha101):
    """
    (-1 * rank(covariance(rank(close), rank(volume), 5)))
    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        # self.data = (-1 * cs_rank(covariance(rank(close), rank(volume), 5)))
        # return self.data
        pass

    def __repr__(self):
        return "(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"


class Alpha014(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha015(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha016(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha017(WQAlpha101):
    """
     (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    """

    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        adv20 = ts_mean(self.env.Volume, 20)
        self.data = -1 * (cs_rank(ts_rank(self.env.Close, 10)) *
                          cs_rank(ts_delta(ts_delta(self.env.Close, 1), 1)) *
                          cs_rank(ts_rank((self.env.Volume / adv20), 5)))
        return self.data
