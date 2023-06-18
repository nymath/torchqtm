import os
import sys
import numpy as np
import pandas as pd
import pickle

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from torchqtm.configurator import *
from torchqtm.utils import Timer
from torchqtm.utils.warnings import catch_warnings
from torchqtm.alphas.alpha101 import *


class NeutralizePE(op.Fundamental):
    def __init__(self, env):
        super().__init__(env)
        self.lag = op.Parameter(5, required_optim=False, feasible_region=None)

    def forward(self):
        self.data = F.divide(1, self.env.PE)
        self.data = F.winsorize(self.data, 'std', 4)
        self.data = F.normalize(self.data)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        self.data = F.ts_mean(self.data, self.lag)
        return self.data


class Momentum01(op.Momentum):
    def __init__(self, env):
        super().__init__(env)

    # @ContextManager(catch_warnings(), self.catch_warnings)
    def forward(self):
        self.data = self.env.Close
        self.data = F.winsorize(self.data, 'std', 4)
        self.data = F.normalize(self.data)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        # self.data = F.ts_returns(self.data, 1)
        self.data = F.ts_delta(self.env.Close, 3)
        return self.data


class Momentum02(op.Momentum):
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        self.data = self.env.Close
        self.data = F.winsorize(self.data, 'std', 4)
        self.data = F.normalize(self.data)
        cond = F.geq(F.ts_mean(self.env.Close, 1), F.ts_mean(self.env.Close, 4))
        self.data = F.trade_when(cond, F.ts_delta(self.env.Close, 3), False)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        return self.data


class Ross(op.Volatility):
    def __init__(self, env):
        super().__init__(env)

    def forward(self):
        raw_shape = self.env.Open.shape
        Open = np.array(F.log(self.env.Open), dtype=np.float64).reshape(*raw_shape, 1)
        High = np.array(F.log(self.env.High), dtype=np.float64).reshape(*raw_shape, 1)
        Low = np.array(F.log(self.env.Low), dtype=np.float64).reshape(*raw_shape, 1)
        Close = np.array(F.log(self.env.Close), dtype=np.float64).reshape(*raw_shape, 1)
        Closel1 = np.array(F.ts_delay(self.env.Close, 1), dtype=np.float64).reshape(*raw_shape, 1)
        data = np.concatenate([Open, High, Low, Close, Closel1], axis=2)

        def aux_func(data_slice):
            cl = {
                'Open': 0,
                'High': 1,
                'Low': 2,
                'Close': 3,
                'Closel1': 4
            }
            return np.sqrt(np.nanmean(0.5 * (data_slice[..., cl['High']] - data_slice[..., cl['Low']]) ** 2, axis=0) -
                           (2 * np.log(2) - 1) * np.nanmean(
                (data_slice[..., cl['Close']] - data_slice[..., cl['Open']]) ** 2, axis=0) +
                           np.nanmean((data_slice[..., cl['Open']] - data_slice[..., cl['Closel1']]) ** 2, axis=0))

        self.data = F.ts_apply(data, 30, aux_func)
        self.data = F.winsorize(self.data, 'std', 4)
        self.data = F.normalize(self.data)
        self.data = pd.DataFrame(self.data, index=self.env.Close.index, columns=self.env.Close.columns)
        cond = F.geq(F.ts_mean(self.close, 5), F.ts_mean(self.close, 22))
        # self.data = F.trade_when(cond, self.data, False)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        return self.data


if __name__ == '__main__':

    def load_data():
        with open(f"{BASE_DIR}/largedata/stocks_f64.pkl", "rb") as f:
            return pickle.load(f)

    dfs = load_data()

    start = '20170101'
    end = '20230101'
    rebalance_alpha = Daily(start, end)
    rebalance_backtest = Weekly(start, end, [-1])
    benchmark = BenchMark('000852.SH', start, end)
    universe = StaticUniverse(IndexComponents('000852.SH', start).data)
    # universe = StaticUniverse(list(dfs['Close'].columns))

    # Create the backtest environment
    alphaEnv = BackTestEnv(dfs=dfs,
                           dates=rebalance_alpha.data,
                           symbols=universe.data)

    btEnv = BackTestEnv(dfs=dfs,
                        dates=rebalance_backtest.data,
                        symbols=universe.data)

    # Create alpha
    alpha = Alpha015(env=alphaEnv)

    with Timer():
        with catch_warnings():
            alpha.forward()
    # run backtest
    bt = GroupTester01(env=btEnv,
                       n_groups=5,
                       weighting='equal',
                       exclude_suspended=False,
                       exclude_limits=False)

    with Timer():
        bt.run_backtest(bt.env.match(alpha.data))
    bt.plot()


