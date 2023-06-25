import torchqtm.op as op
import torchqtm.op.functional as F
from torchqtm.op.functional import *
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from torchqtm.tdbt.backtest import BackTestEnv
from torchqtm.base import BaseAlpha


class MLAlpha(BaseAlpha):
    def __init__(self, env: BackTestEnv, model=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.type = 'machine learning alpha'
        self.model = model

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        """assign self.rawdata and return"""
        raise NotImplementedError




