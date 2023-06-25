import torchqtm.op as op
import torchqtm.op.functional as F
from torchqtm.op.functional import *
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from torchqtm.tdbt.backtest import BackTestEnv
from torchqtm.base import BaseAlpha


# __all__ = [f"Alpha{str(i).zfill(3)}" for i in range(1, 102)]
# https://github1s.com/yli188/WorldQuant_alpha101_code/blob/master/101Alpha_code_1.py#L378
# https://chat.openai.com/share/7147e5ec-d1f9-4282-9394-c59fd35ee13b
class WQAlpha101(BaseAlpha, metaclass=ABCMeta):
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
    -1 * correlation(rank(open), rank(volume), 10)
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
    ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    """

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


class Alpha010(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        pass

    def forward(self):
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
    """
               [rank]
                 |
               [mul]
               /   \
           [mul]   [sub]
           /   \     /   \
        [mul] vwap high close
         /   \
       [mul] adv20
       /   \
      -1 returns

    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"

    def forward(self):
        self.data = cs_rank(((((-1 * self.returns) * self.adv20) * self.vwap) * (self.high - self.close)))
        return self.data


class Alpha026(WQAlpha101):
    """
        [mul]
        /   \
      -1  [ts_max]
          /   \
    [correlation] 3
     /   |   \
    [ts_rank] [ts_rank] 5
     /   \    /   \
    volume 5 high  5

    """

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
    """
                   [div]
                   /   \
               [mul]   [sum]
               /   \       /   \
          [sub]    [sum] volume 20
          /   \       /   \
        1.0 [rank] volume 5
             |
           [add]
           /   \
       [add]  [sign]
       /   \    |
    [sign] [sign] [delay]
     /   \   /   \  /   \
    [delay] [delay] close 3
     /   \    /   \
    close 1 close 2
    """

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


# TODO: implement it
class Alpha031(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        return "((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + " \
               "rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))"

    def forward(self):
        pass


# TODO: implement it
class Alpha032(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        return "(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha033(WQAlpha101):
    """
             [rank]
             |
            [mul]
            /   \
          -1    [pow]
                /   \
             [sub]   1
             /   \
            1   [div]
                /   \
             open close
    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "rank((-1 * ((1 - (open / close))^1)))"

    def forward(self):
        self.data = cs_rank(-1 + (self.open / self.close))
        return self.data


class Alpha034(WQAlpha101):
    """
             [rank]
             |
           [add]
           /   \
       [sub]   [sub]
       /   \     /   \
      1  [rank] 1  [rank]
         |         |
      [div]    [delta]
     /   \     /   \
    [stddev] [stddev] close 1
     /   \     /   \
    returns 2 returns 5

    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"

    def forward(self):
        temp = ts_std_dev(self.returns, 2) / ts_std_dev(self.returns, 5)
        self.data = cs_rank(2 - cs_rank(temp) - cs_rank(ts_delta(self.close, 1)))
        return self.data


class Alpha035(WQAlpha101):
    """
              [mul]
              /   \
           [mul]  [sub]
           /   \     / \
     [ts_Rank] [sub] 1 [ts_Rank]
     /   \     /     /   \
    volume 32 1 [ts_Rank] returns 32
               | /   \
             [sub]   16
             / | \
         [add] low
         /   \
       close high
    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))"

    def forward(self):
        self.data = ((ts_rank(self.volume, 32) *
                      (1 - ts_rank(self.close + self.high - self.low, 16))) *
                     (1 - ts_rank(self.returns, 32)))
        return self.data


class Alpha036(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False  # It seems too complex and use too long data.

    def __repr__(self):
        return "(((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + " \
               "(0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 " \
               "* rank((((sum(close, 200) / 200) - open) * (close - open)))))"

    def forward(self):
        pass


class Alpha037(WQAlpha101):
    """
              [add]
              /   \
        [rank]   [rank]
        /          |
    [correlation] [sub]
     /   |   \     /   \
    [delay] close 200 open close
     /   \
    [sub] 1
     /   \
    open close
    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"

    def forward(self):
        self.data = cs_rank(ts_corr(ts_delay(self.open - self.close, 1), self.close, 200)) \
                    + cs_rank(self.open - self.close)
        return self.data


class Alpha038(WQAlpha101):
    """
              (*)
              / \
             /   \
            /     \
        [mul]    [cs_rank]
         /  \        |
        -1  [cs_rank] [divide]
            |        /   \
         [ts_rank] self.close self.open
            |
         [self.close, 10]
    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))"

    def forward(self):
        self.data = (-1 * cs_rank(ts_rank(self.close, 10))) * cs_rank(divide(self.close, self.open))
        return self.data


class Alpha039(WQAlpha101):
    """
            [mul]
            /   \
         [mul]  [add]
         /   \     / \
       -1  [rank] 1 [rank]
            |         |
       [mul]     [sum]
       /   \         |
    [delta] [sub] [returns, 250]
     /   \   /   \
    close 7  1  [rank]
               |
          [decay_linear]
           /        \
      [div]         9
     /   \
    volume adv20
    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(" \
               "returns, 250))))"

    def forward(self):
        self.data = ((-1 * cs_rank(
            ts_delta(self.close, 7) * (1 - cs_rank(ts_decay_linear((self.volume / self.adv20), 9).CLOSE)))) *
                     (1 + cs_rank(ts_mean(self.returns, 250))))
        return self.data


class Alpha040(WQAlpha101):
    """
            [mul]
            /   \
        [mul]  [correlation]
        /   \     /   |   \
      -1  [rank] high volume 10
          |
       [stddev]
       /   \
     high  10
    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"

    def forward(self):
        self.data = -1 * cs_rank(ts_std_dev(self.high, 10)) * ts_corr(self.high, self.volume, 10)
        return self.data


class Alpha041(WQAlpha101):
    """
           [sub]
           /   \
       [pow]   vwap
       /   \
     [mul]  0.5
     /   \
    high low
    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(((high * low)^0.5) - vwap)"

    def forward(self):
        self.data = pow((self.high * self.low), 0.5) - self.vwap
        return self.data


class Alpha042(WQAlpha101):
    """
        [div]
        /   \
    [rank]  [rank]
      |       |
    [sub]   [add]
    /   \   /   \
    vwap close vwap close
    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(rank((vwap - close)) / rank((vwap + close)))"

    def forward(self):
        self.data = cs_rank((self.vwap - self.close)) / cs_rank((self.vwap + self.close))
        return self.data


class Alpha043(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)"

    def forward(self):
        self.data = ts_rank(self.volume / self.adv20, 20) * ts_rank((-1 * ts_delta(self.close, 7)), 8)
        return self.data


class Alpha044(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(-1 * correlation(high, rank(volume), 5))"

    def forward(self):
        self.data = ts_corr(self.high, cs_rank(self.volume), 5)
        return self.data


class Alpha045(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(" \
               "close, 5), sum(close, 20), 2))))"

    def forward(self):
        df = ts_corr(self.close, self.volume, 2)
        self.data = -1 * (cs_rank(ts_mean(ts_delay(self.close, 5), 20)) * df *
                          cs_rank(ts_corr(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))
        return self.data


class Alpha046(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        return "((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * " \
               "1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :(" \
               "(-1 * 1) * (close - delay(close, 1)))))"

    def forward(self):
        # inner = ((ts_delay(self.close, 20) - ts_delay(self.close, 10)) / 10) - ((ts_delay(self.close, 10) - self.close) / 10)
        # alpha = (-1 * ts_delta(self.close))
        # alpha[inner < 0] = 1
        # alpha[inner > 0.25] = -1
        # self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha047(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - " \
               "rank((vwap - delay(vwap, 5))))"

    def forward(self):
        self.data = ((((cs_rank((1 / self.close)) * self.volume) / self.adv20) *
                      ((self.high * cs_rank((self.high - self.close))) / (ts_mean(self.high, 5) / 5)))
                     - cs_rank((self.vwap - ts_delay(self.vwap, 5))))
        return self.data


class Alpha048(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        return "(indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, " \
               "1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha049(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 " \
               ": ((-1 * 1) * (close - delay(close, 1))))"

    def forward(self):
        condition = (((ts_delay(self.close, 20) - ts_delay(self.close, 10)) / 10) - (
                (ts_delay(self.close, 10) - self.close) / 10)) < -0.1
        value_if_true = 1
        value_if_false = -1 * ts_delta(self.close, 1)
        self.data = if_else(condition, value_if_true, value_if_false)
        return self.data


class Alpha050(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"

    def forward(self):
        self.data = (-1 * ts_max(cs_rank(ts_corr(cs_rank(self.volume), cs_rank(self.vwap), 5)), 5))
        return self.data


class Alpha051(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? " \
               "1 : ((-1 * 1) * (close - delay(close, 1))))"

    def forward(self):
        condition = (((ts_delay(self.close, 20) - ts_delay(self.close, 10)) / 10) - (
                (ts_delay(self.close, 10) - self.close) / 10)) < -0.05
        value_if_true = 1
        value_if_false = -1 * ts_delta(self.close, 1)
        self.data = if_else(condition, value_if_true, value_if_false)
        return self.data


class Alpha052(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, " \
               "20)) / 220))) * ts_rank(volume, 5))"

    def forward(self):
        self.data = (((-1 * ts_delta(ts_min(self.low, 5), 5)) *
                      cs_rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume,
                                                                                                         5))
        return self.data


class Alpha053(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        return "(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"

    def forward(self):
        inner = (self.close - self.low)
        self.data = -1 * ts_delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)
        return self.data


# Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
def alpha054(self):
    inner = (self.low - self.high).replace(0, -0.0001)
    return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))


class Alpha054(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha055(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), " \
               "rank(volume), 6))"

    def forward(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / divisor
        self.data = ts_corr(cs_rank(inner), cs_rank(self.volume), 6)
        return self.data


class Alpha056(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        return "(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"

    def forward(self):
        return self.data


class Alpha057(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"

    def forward(self):
        self.data = (self.close - self.vwap) / ts_decay_linear(cs_rank(ts_arg_max(self.close, 30)), 2)
        return self.data

# Alpha#58	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))

# Alpha#59	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))


class Alpha060(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        return "(0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(" \
               "rank(ts_argmax(close, 10))))))"

    def forward(self):
        divisor = (self.high - self.low) + 0.001
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        # self.data = - ((2 * ts_scale(cs_rank(inner))) - scale(rank(ts_argmax(self.close, 10))))
        return self.data


class Alpha061(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))"

    def forward(self):
        adv180 = ts_mean(self.volume, 180)
        self.data = (cs_rank((self.vwap - ts_min(self.vwap, 16))) < cs_rank(ts_corr(self.vwap, adv180, 18)))
        return self.data


class Alpha062(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((" \
               "high + low) / 2)) + rank(high))))) * -1)"

    def forward(self):
        adv20 = ts_mean(self.volume, 20)
        self.data = ((cs_rank(ts_corr(self.vwap, ts_mean(adv20, 22), 10)) < cs_rank(
            ((cs_rank(self.open) + cs_rank(self.open)) < (cs_rank(((self.high + self.low) / 2)) + cs_rank(self.high))))) * -1)
        return self.data


# Alpha#63	 ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)
class Alpha063(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        return "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha064(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), " \
               "16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)"

    def forward(self):
        adv120 = ts_mean(self.volume, 120)
        self.data = ((cs_rank(
            ts_corr(ts_mean(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13), ts_mean(adv120, 13), 17)) < cs_rank(
            ts_delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))), 3.69741))) * -1)
        return self.data


class Alpha065(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < " \
               "rank((open - ts_min(open, 13.635)))) * -1)"

    def forward(self):
        adv60 = ts_mean(self.volume, 60)
        self.data = ((cs_rank(
            ts_corr(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), ts_mean(adv60, 9), 6)) < cs_rank(
            (self.open - ts_min(self.open, 14)))) * -1)
        return self.data


# # Alpha#66	 ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
# def alpha066(self):
#     return ((rank(decay_linear(delta(self.vwap, 4).to_frame(), 7).CLOSE) + ts_rank(decay_linear(((((
#                                                                                                            self.low * 0.96633) + (
#                                                                                                            self.low * (
#                                                                                                            1 - 0.96633))) - self.vwap) / (
#                                                                                                          self.open - (
#                                                                                                          (
#                                                                                                                  self.high + self.low) / 2))).to_frame(),
#                                                                                                 11).CLOSE, 7)) * -1)


# Alpha#67	 ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
class Alpha067(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False # 无数据

    def __repr__(self):
        return "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha068(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (" \
               "low * (1 - 0.518371))), 1.06157))) * -1)"

    def forward(self):
        adv15 = ts_mean(self.volume, 15)
        self.data = ((ts_rank(ts_corr(cs_rank(self.high), cs_rank(adv15), 9), 14) < cs_rank(
            ts_delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1.06157))) * -1)
        return self.data



# Alpha#69	 ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)

# Alpha#70	 ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)

# Alpha#71	 max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
def alpha071(self):
    adv180 = sma(self.volume, 180)
    p1 = ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18).to_frame(), 4).CLOSE, 16)
    p2 = ts_rank(
        decay_linear((rank(((self.low + self.open) - (self.vwap + self.vwap))).pow(2)).to_frame(), 16).CLOSE, 4)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df.at[df['p1'] >= df['p2'], 'max'] = df['p1']
    df.at[df['p2'] >= df['p1'], 'max'] = df['p2']
    return df['max']
    # return max(ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18).to_frame(), 4).CLOSE, 16), ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)).to_frame(), 16).CLOSE, 4))


class Alpha071(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)
        self.Enabled = False

    def __repr__(self):
        return "max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), " \
               "4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data


class Alpha072(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(" \
               "correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))"

    def forward(self):
        adv40 = ts_mean(self.volume, 40)
        numerator = cs_rank(ts_decay_linear(ts_corr((self.high + self.low)/2, adv40, 9), 10))
        denominator = cs_rank(ts_decay_linear(ts_corr(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3))
        self.data = numerator / denominator
        return self.data


# Alpha#73	 (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
def alpha073(self):
    p1 = rank(decay_linear(delta(self.vwap, 5).to_frame(), 3).CLOSE)
    p2 = ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / (
            (self.open * 0.147155) + (self.low * (1 - 0.147155)))) * -1).to_frame(), 3).CLOSE, 17)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df.at[df['p1'] >= df['p2'], 'max'] = df['p1']
    df.at[df['p2'] >= df['p1'], 'max'] = df['p2']
    return -1 * df['max']
    # return (max(rank(decay_linear(delta(self.vwap, 5).to_frame(), 3).CLOSE),ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open *0.147155) + (self.low * (1 - 0.147155)))) * -1).to_frame(), 3).CLOSE, 17)) * -1)

class Alpha073(WQAlpha101):
    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"

    def forward(self):
        self.data = -1 * cs_rank(ts_delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)
        return self.data
# Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
def alpha074(self):
    adv30 = sma(self.volume, 30)
    return ((rank(correlation(self.close, sma(adv30, 37), 15)) < rank(
        correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11))) * -1)


# Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
def alpha075(self):
    adv50 = sma(self.volume, 50)
    return (rank(correlation(self.vwap, self.volume, 4)) < rank(correlation(rank(self.low), rank(adv50), 12)))


# Alpha#76	 (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)

# Alpha#77	 min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
def alpha077(self):
    adv40 = sma(self.volume, 40)
    p1 = rank(
        decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)).to_frame(), 20).CLOSE)
    p2 = rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3).to_frame(), 6).CLOSE)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df.at[df['p1'] >= df['p2'], 'min'] = df['p2']
    df.at[df['p2'] >= df['p1'], 'min'] = df['p1']
    return df['min']
    # return min(rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)).to_frame(), 20).CLOSE),rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3).to_frame(), 6).CLOSE))


# Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
def alpha078(self):
    adv40 = sma(self.volume, 40)
    return (rank(
        correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20), ts_sum(adv40, 20), 7)).pow(
        rank(correlation(rank(self.vwap), rank(self.volume), 6))))


# Alpha#79	 (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))

# Alpha#80	 ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)

# Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
def alpha081(self):
    adv10 = sma(self.volume, 10)
    return ((rank(log(product(rank((rank(correlation(self.vwap, ts_sum(adv10, 50), 8)).pow(4))), 15))) < rank(
        correlation(rank(self.vwap), rank(self.volume), 5))) * -1)


# Alpha#82	 (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)

# Alpha#83	 ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
def alpha083(self):
    return ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) / (
            ((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))


# Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
def alpha084(self):
    return pow(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close, 5))


# Alpha#85	 (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
def alpha085(self):
    adv30 = sma(self.volume, 30)
    return (rank(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30, 10)).pow(
        rank(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10), 7))))


# Alpha#86	 ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)

def alpha086(self):
    adv20 = sma(self.volume, 20)
    return ((ts_rank(correlation(self.close, sma(adv20, 15), 6), 20) < rank(
        ((self.open + self.close) - (self.vwap + self.open)))) * -1)


# Alpha#87	 (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)

# Alpha#88	 min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
def alpha088(self):
    adv60 = sma(self.volume, 60)
    p1 = rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))).to_frame(),
                           8).CLOSE)
    p2 = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8).to_frame(), 7).CLOSE, 3)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df.at[df['p1'] >= df['p2'], 'min'] = df['p2']
    df.at[df['p2'] >= df['p1'], 'min'] = df['p1']
    return df['min']
    # return min(rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))).to_frame(),8).CLOSE), ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60,20.6966), 8).to_frame(), 7).CLOSE, 3))


# Alpha#89	 (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))

# Alpha#90	 ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)

# Alpha#91	 ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)

# Alpha#92	 min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
def alpha092(self):
    adv30 = sma(self.volume, 30)
    p1 = ts_rank(
        decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).to_frame(), 15).CLOSE,
        19)
    p2 = ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8).to_frame(), 7).CLOSE, 7)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df.at[df['p1'] >= df['p2'], 'min'] = df['p2']
    df.at[df['p2'] >= df['p1'], 'min'] = df['p1']
    return df['min']
    # return  min(ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).to_frame(), 15).CLOSE,19), ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8).to_frame(), 7).CLOSE,7))


# Alpha#93	 (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))

# Alpha#94	 ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
def alpha094(self):
    adv60 = sma(self.volume, 60)
    return ((rank((self.vwap - ts_min(self.vwap, 12))).pow(
        ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)) * -1))


# Alpha#95	 (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
def alpha095(self):
    adv40 = sma(self.volume, 40)
    return (rank((self.open - ts_min(self.open, 12))) < ts_rank(
        (rank(correlation(sma(((self.high + self.low) / 2), 19), sma(adv40, 19), 13)).pow(5)), 12))


# Alpha#96	 (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
def alpha096(self):
    adv60 = sma(self.volume, 60)
    p1 = ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume).to_frame(), 4), 4).CLOSE, 8)
    p2 = ts_rank(
        decay_linear(ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4), 13).to_frame(), 14).CLOSE,
        13)
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    df.at[df['p1'] >= df['p2'], 'max'] = df['p1']
    df.at[df['p2'] >= df['p1'], 'max'] = df['p2']
    return -1 * df['max']
    # return (max(ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume).to_frame(), 4),4).CLOSE, 8), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7),ts_rank(adv60, 4), 4), 13).to_frame(), 14).CLOSE, 13)) * -1)


# Alpha#97	 ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)

# Alpha#98	 (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
def alpha098(self):
    adv5 = sma(self.volume, 5)
    adv15 = sma(self.volume, 15)
    return (rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5).to_frame(), 7).CLOSE) - rank(
        decay_linear(ts_rank(ts_argmin(correlation(rank(self.open), rank(adv15), 21), 9), 7).to_frame(), 8).CLOSE))


class Alpha099(WQAlpha101):
    """
               [mul]
               /   \
           [lt]   -1
           /   \
       [rank] [rank]
       /        |
    [correlation] [correlation]
     /   |   \    /   |   \
    [sum] [sum] 8.8136 low volume 6.28259
     /   \    /   \
    [div] 19.8975 adv60 19.8975
     /   \
    high low

    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(" \
               "low, volume, 6.28259))) * -1)"

    def forward(self):
        adv60 = ts_mean(self.volume, 60)
        self.data = (cs_rank(ts_corr(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)) <
                     cs_rank(ts_corr(self.low, self.volume, 6))) * -1
        return self.data


class Alpha101(WQAlpha101):
    """
           [div]
           /   \
       [sub]  [add]
       /   \   /   \
     close open [sub] .001
               /   \
             high  low
    """

    def __init__(self, env):
        super().__init__(env)

    def __repr__(self):
        return "((close - open) / ((high - low) + .001))"

    def forward(self):
        self.data = (self.close - self.open) / ((self.high - self.low) + 0.001)
        return self.data
