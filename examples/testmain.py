import os
import sys
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from torchqtm.utils.rebalance import Weekly
from torchqtm.utils.universe import StaticUniverse, IndexComponents
from torchqtm.utils.benchmark import BenchMark
from torchqtm.tdbt.backtest import BackTestEnv
import torchqtm.op as op
import torchqtm.op.functional as F
from torchqtm.utils import Timer
import pickle
import _C._functional as CF

start = '20170101'
end = '20230101'
rebalance = Weekly(start, end, [-1])
benchmark = BenchMark('000905.SH', start, end)
universe = StaticUniverse(IndexComponents('000905.SH', start).data)


class NeutralizePE(op.Fundamental):
    def __init__(self, env):
        super().__init__(env)

    def forward(self, factor):
        self.data = F.divide(1, factor)
        # self.rawdata = F.winsorize(self.rawdata, 'std', 4)
        # self.rawdata = F.normalize(self.rawdata)
        # self.rawdata = F.group_neutralize(self.rawdata, self.env.Sector)
        # self.rawdata = F.regression_neut(self.rawdata, self.env.MktVal)
        with Timer():
            F.regression_neut(self.data, self.env.Sector)
        with Timer():
            CF.regression_neut(self.data, self.env.Sector)
        return self.data


if __name__ == '__main__':
    # Load the rawdata
    with open("examples/largedata/Stocks.pkl", "rb") as f:
        dfs = pickle.load(f)
    # Create the backtest environment
    btEnv = BackTestEnv(dfs=dfs,
                        dates=rebalance.rebalance_dates,
                        symbols=universe.symbols)
    # Create alpha
    alphas = NeutralizePE(env=btEnv)
    alphas.forward(btEnv.match(dfs['PE']))
    # run backtest
    # bt = QuickBackTesting01(env=btEnv,
    #                         universe=universe,
    #                         n_groups=10)
    # bt.run_backtest(alphas.rawdata)
    #
    # #
    # # from torchqtm.tdbt.stats import ic
    # # icSeries = ic(alphas.rawdata, BtEnv._FutureReturn, method='spearman')
    #
    # # plot the result
    # fig = plt.figure(figsize=(20, 12))
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # color_generator = ColorGenerator(10)
    # colors = color_generator()
    # for i in range(10):
    #     ax.plot((1+bt.returns.iloc[:, i]).cumprod(), label=f'group_{i+1}', color=colors[i])
    # # temp = benchmark.rawdata.loc[rebalance.rebalance_dates]['Close'].pct_change()
    # # temp.fillna(0, inplace=True)
    # # ax.plot(temp.cumsum(), label='benchmark', color='blue')
    # # ax.plot((bt.returns['group_10']).cumsum()-temp.cumsum(), label='excess', color='orange')
    # plt.legend(fontsize=16)
    # plt.show()


