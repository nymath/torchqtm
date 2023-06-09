import torchqtm.op as op
import torchqtm.op.functional as F
from torchqtm.op.functional import *
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from torchqtm.vbt.backtest import BackTestEnv
from torchqtm.op.base import BaseAlpha


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


class Alpha004(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = -1 * ts_rank(cs_rank(self.low), 9)
        return self.data


class Alpha008(WQAlpha101):
    """
    (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    """

    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = -1 * (cs_rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                                   ts_delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))
        return self.data


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


class Alpha012(WQAlpha101):
    """
    (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    """
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = sign(ts_delta(self.volume, 1)) * (-1 * ts_delta(self.close, 1))
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
