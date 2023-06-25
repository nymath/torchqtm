import torchqtm.op as op
import torchqtm.op.functional as F
from torchqtm.op.functional import *
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from torchqtm.tdbt.backtest import BackTestEnv
from torchqtm.base import BaseAlpha


class IntroductoryAlpha(BaseAlpha, metaclass=ABCMeta):
    def __init__(self, env: BackTestEnv, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'IntroductoryAlpha'

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError


class Step01(IntroductoryAlpha):
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        adv20 = ts_mean(self.volume, 20)
        event = geq(self.volume, adv20)
        alpha_if_true = 2*(-ts_delta(self.close, 3))
        alpha_if_false = (-ts_delta(self.close, 3))
        self.data = if_else(event, alpha_if_true, alpha_if_false)
        return self.data


class Step02(IntroductoryAlpha):
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        adv20 = ts_mean(self.volume, 20)
        event = geq(self.volume, adv20)
        alpha = (-ts_delta(self.close, 3))
        self.data = trade_when(event, alpha, False)
        return self.data

