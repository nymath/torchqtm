import os
import sys

from torchqtm.utils.rebalance import Weekly
from torchqtm.utils.visualization import ColorGenerator
from torchqtm.utils.universe import StaticUniverse, IndexComponents
from torchqtm.utils.benchmark import BenchMark
from torchqtm.vbt.backtest import BackTestEnv, QuickBackTesting01
import torchqtm.op as op
import numpy as np
import torchqtm.op.functional as F

import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)


start = '20170101'
end = '20230101'
rebalance = Weekly(start, end, [-1])
benchmark = BenchMark('000905.SH', start, end)
universe = StaticUniverse(IndexComponents('000905.SH', start).data)


class NeutralizePE(op.Fundamental):
    def __init__(self, env):
        super().__init__(env)

    def operate(self, factor):
        self.data = F.divide(1, factor)
        self.data = self.data.astype(np.float64)
        self.data = F.winsorize(self.data, 'std', 4)
        self.data = F.normalize(self.data)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        self.data = F.ts_mean(self.data, 5)
        return self.data


class Momentum01(op.Momentum):
    def __init__(self, env):
        super().__init__(env)

    def operate(self):
        self.data = F.divide(1, self.env.Close)
        self.data = self.data.astype(np.float64)
        self.data = F.winsorize(self.data, 'std', 4)
        self.data = F.normalize(self.data)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        self.data = self.data.astype(np.float64)
        # self.data = F.ts_returns(self.data, 1)
        self.data = F.ts_delta(self.env.Close, 3)
        return self.data


class Momentum02(op.Momentum):
    def __init__(self, env):
        super().__init__(env)

    def operate(self):
        self.env.Close = self.env.Close.astype(np.float64)
        self.data = self.env.Close
        self.data = F.winsorize(self.data, 'std', 4)
        self.data = F.normalize(self.data)
        cond = F.geq(F.ts_mean(self.env.Close, 1), F.ts_mean(self.env.Close, 4))
        self.data = F.trade_when(cond, F.ts_delta(self.env.Close, 3), False)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        return self.data


if __name__ == '__main__':
    # Load the rawdata
    with open(f"{BASE_DIR}/largedata/Stocks.pkl", "rb") as f:
        dfs = pickle.load(f)
    # Create the backtest environment
    btEnv = BackTestEnv(dfs=dfs,
                        dates=rebalance.rebalance_dates,
                        symbols=universe.symbols)
    # Create alpha
    # alphas = NeutralizePE(env=btEnv)
    alphas = Momentum02(env=btEnv)
    # alphas.operate(btEnv.match_env(dfs['PE']))
    alphas.operate()
    # run backtest
    bt = QuickBackTesting01(env=btEnv,
                            universe=universe,
                            n_groups=10)
    bt.run_backtest(alphas.data)

    # plot the result
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    color_generator = ColorGenerator(10)
    colors = color_generator()
    for i in range(10):
        ax.plot((1+bt.returns.iloc[:, i]).cumprod(), label=f'group_{i+1}', color=colors[i])
    fig.legend(fontsize=16)
    fig.show()


