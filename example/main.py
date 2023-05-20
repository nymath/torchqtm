from torchquantum.quant.utils import Calendar
from torchquantum.quant.visualization import ColorGenerator
from torchquantum.quant.universe import StaticUniverse
from torchquantum.quant.backtest import BackTestEnv, QuickBackTesting01

import torchquantum.quant.op as op
import torchquantum.quant.op.functional as F
import matplotlib.pyplot as plt
import time
import pickle

tt0 = time.time()
testdata = {}
calendar = Calendar('20100101', '20170101')
trade_dates = calendar.trade_dates
_ = calendar.create_weekly_groups()
rebalance_dates = [x[-1] for x in _.values()]

with open("./example/data/adata.pkl", "rb") as f:
    dfs = pickle.load(f)

t0 = time.time()

BtEnv = BackTestEnv(dfs=dfs,
                    rebalance_dates=dfs['Close'].index,
                    symbols=dfs['Close'].columns)


class NeutralizePE(op.Fundamental):
    def __init__(self, env, factor):
        super().__init__(env)
        self.factor = self.env.match_env(factor)

    def operate(self, *args, **kwargs):
        self.data = self.factor
        self.data = F.winsorize(self.data, 0.05)
        self.data = F.normalize(self.data)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        return self.data


alphas = NeutralizePE(BtEnv, dfs['factor'])
alphas.operate()

bt = QuickBackTesting01(env=BtEnv,
                        universe=StaticUniverse(dfs['Close'].columns),
                        n_groups=10)

bt.run_backtest(alphas.data)

fig = plt.figure(figsize=(20, 8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

color_generator = ColorGenerator(10)
colors = color_generator()

for i in range(10):
    ax.plot((1+bt.returns.iloc[:, i]).cumprod(), label=str(i), color=colors[i])
plt.show()

print(time.time() - t0)
print("Hello world!")

