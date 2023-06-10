import torchqtm.op as op
import torchqtm.op.functional as F
from torchqtm.op.functional import *
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from torchqtm.vbt.backtest import BackTestEnv
from torchqtm.op.base import BaseAlpha

# __all__ = [f"Alpha{str(i).zfill(3)}" for i in range(1, 102)]
# https://github1s.com/yli188/WorldQuant_alpha101_code/blob/master/101Alpha_code_1.py#L378


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
        self.data = -1 * ts_corr(cs_rank(ts_delta(log(self.volume), 2)), cs_rank((self.close - self.open) / self.open),
                                 6)
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
        self.data = -1 * ts_corr(cs_rank(ts_delta(log(self.volume), 2)), cs_rank((self.close - self.open) / self.open),
                                 6)
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


class Alpha018(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))"

    def forward(self):
        df = ts_corr(self.close, self.open, 10)
        self.data = -1 * (cs_rank((ts_std_dev(abs(self.close - self.open), 5) + (self.close - self.open)) +
                                  df))
        return self.data


class Alpha019(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))"

    def forward(self):
        self.data = ((-1 * sign((self.close - ts_delay(self.close, 7)) + ts_delta(self.close, 7))) *
                     (1 + cs_rank(1 + ts_sum(self.returns, 250))))
        return self.data


class Alpha020(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))"

    def forward(self):
        self.data = -1 * (cs_rank(self.open - ts_delay(self.high, 1)) *
                          cs_rank(self.open - ts_delay(self.close, 1)) *
                          cs_rank(self.open - ts_delay(self.low, 1)))
        return self.data


# class Alpha021(WQAlpha101):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def __repr__(self):
#         return "((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close," \
#                "2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume " \
#                "/adv20) == 1)) ? 1 : (-1 * 1))))"
#
#     def forward(self):
#         cond_1 = ts_mean(self.close, 8) + ts_std_dev(self.close, 8) < ts_mean(self.close, 2)
#         cond_2 = ts_mean(self.volume, 20) / self.volume < 1
#         condition =
#         self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
#         return self.data


class Alpha022(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"

    def forward(self):
        df = ts_corr(self.high, self.volume, 5)
        self.data = -1 * ts_delta(df, 5) * cs_rank(ts_std_dev(self.close, 20))
        return self.data


class Alpha023(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"

    def forward(self):
        condition = less(ts_mean(self.high, 20), self.high)
        value_if_true = -1 * ts_delta(self.high, 2)
        value_if_false = 0
        self.data = if_else(condition, value_if_true, value_if_false)
        return self.data


class Alpha024(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, " \
               "100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(" \
               "close, 3)))"

    def forward(self):
        condition = ts_delta(ts_mean(self.close, 100), 100) / geq(ts_delay(self.close, 100), 0.05)
        value_if_true = -1 * (self.close - ts_min(self.close, 100))
        value_if_false = -1 * ts_delta(self.close, 3)
        self.data = if_else(condition, value_if_true, value_if_false)
        return self.data


class Alpha025(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"

    def forward(self):
        self.data = cs_rank(((((-1 * self.returns) * self.adv20) * self.vwap) * (self.high - self.close)))
        return self.data


class Alpha026(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"

    def forward(self):
        df = ts_corr(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        self.data = -1 * ts_max(df, 3)
        return self.data


class Alpha027(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)"

    def forward(self):
        condition = (0.5 < cs_rank((ts_mean(ts_corr(cs_rank(self.volume), cs_rank(self.vwap), 6), 2))))
        value_if_true = -1
        value_if_false = 1
        self.data = if_else(condition, value_if_true, value_if_false)
        return self.data


# class Alpha028(WQAlpha101):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def __repr__(self):
#         return "scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"
#
#     def forward(self):
#         self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
#         return self.data


# class Alpha029(WQAlpha101):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def __repr__(self):
#         return "(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), " \
#                "1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))"
#
#     def forward(self):
#         self.data = (ts_min(cs_rank(cs_rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
#                 ts_rank(delay((-1 * self.returns), 6), 5))
#         return self.data


class Alpha030(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((" \
               "delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))"

    def forward(self):
        delta_close = ts_delta(self.close, 1)
        inner = sign(delta_close) + sign(ts_delay(delta_close, 1)) + sign(ts_delay(delta_close, 2))
        self.data = ((1.0 - cs_rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)
        return self.data


class Alpha101(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((close - open) / ((high - low) + .001))"

    def forward(self):
        self.data = (self.close - self.open) /((self.high - self.low) + 0.001)
        return self.data


