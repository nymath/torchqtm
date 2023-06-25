import torchqtm.op as op
import torchqtm.op.functional as F
from torchqtm.op.functional import *
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from torchqtm.tdbt.backtest import BackTestEnv
from torchqtm.base import BaseAlpha


class ChatGPT(BaseAlpha, metaclass=ABCMeta):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'WorldQuant Alpha101'

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError


class Alpha001(ChatGPT):
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = cs_rank(ts_sum(F.max(self.high - self.close, self.close - self.low), 5)) / 5
        return self.data


class Alpha002(ChatGPT):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = True

    def __repr__(self):
        return ""

    def forward(self):
        VWAP = ts_sum(self.volume * self.close, 20) / ts_sum(self.volume, 20)
        self.data = cs_rank(self.close - VWAP) / cs_rank(ts_std_dev(self.close, 20))
        return self.data


class Alpha003(ChatGPT):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return ""

    def forward(self):
        self.data = cs_rank(self.env.MktVal) * cs_rank(self.env.PE)
        return self.data


class Alpha004(ChatGPT):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return ""

    def forward(self):
        self.data = -cs_rank(ts_corr(cs_rank(self.close), cs_rank(ts_mean(self.volume, 10)), 6))
        return self.data
