from quant.bt.rebalance import Calendar
from quant.visualization.visualization import ColorGenerator
from quant.bt.universe import StaticUniverse
from quant.bt.backtest import BackTestEnv, QuickBackTesting01, QuickBackTesting02

import quant.op as op
import quant.op.functional as F
import matplotlib.pyplot as plt
import time
import pickle

tt0 = time.time()
testdata = {}
calendar = Calendar('20170101', '20230101')
trade_dates = calendar.trade_dates
_ = calendar.create_weekly_groups()
rebalance_dates = sorted([x[-1] for x in _.values()])


class Timer(object):
    def __init__(self):
        self.t0 = time.time()

    def tick(self, *args):
        print(time.time() - self.t0, *args)
        self.t0 = time.time()


with open("largedata/Stocks.pkl", "rb") as f:
    dfs = pickle.load(f)


BtEnv = BackTestEnv(dfs=dfs,
                    dates=rebalance_dates,
                    symbols=dfs['Close'].columns)


class NeutralizePE(op.Fundamental):
    def __init__(self, env, factor):
        super().__init__(env)
        self.factor = self.env.match_env(factor)

    def operate(self, *args, **kwargs):

        self.data = 1 / self.factor
        timer.tick()
        self.data = F.winsorize(self.data, 'quantile', 0.05)
        timer.tick('winsorize')
        self.data = F.normalize(self.data)
        timer.tick('normalize')
        self.data = F.group_neutralize(self.data, self.env.Sector)
        timer.tick('group_neutralize')
        self.data = F.regression_neut(self.data, self.env.MktVal)
        timer.tick('regression_neut')
        return self.data


if __name__ == '__main__':
    timer = Timer()

    alphas = NeutralizePE(BtEnv, dfs['pe'])
    alphas.operate()

    bt = QuickBackTesting01(env=BtEnv,
                            universe=StaticUniverse(dfs['Close'].columns),
                            n_groups=10)

    bt.run_backtest(alphas.data)

    timer.tick()

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    color_generator = ColorGenerator(10)
    colors = color_generator()

    for i in range(10):
        ax.plot((1+bt.returns.iloc[:, i]).cumprod(), label=str(i), color=colors[i])
    plt.show()

    print("Hello world!")

