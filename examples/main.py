from quant.vbt.rebalance import Calendar, Weekly
from quant.visualization.visualization import ColorGenerator
from quant.vbt.universe import StaticUniverse, IndexComponents
from quant.vbt.benchmark import BenchMark
from quant.vbt.backtest import BackTestEnv, QuickBackTesting01
import quant.op as op
import quant.op.functional as F
import matplotlib.pyplot as plt
import time
import pickle
tt0 = time.time()
start = '20170101'
end = '20230101'
rebalance = Weekly(start, end, -1)
benchmark = BenchMark('000905.SH', start, end)
universe = StaticUniverse(IndexComponents('000905.SH', start).data)


class NeutralizePE(op.Fundamental):
    def __init__(self, env):
        super().__init__(env)

    def operate(self, factor):
        self.data = F.divide(1, factor)
        self.data = F.winsorize(self.data, 'std', 4)
        self.data = F.normalize(self.data)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        return self.data


if __name__ == '__main__':
    # Load the data
    with open("largedata/Stocks.pkl", "rb") as f:
        dfs = pickle.load(f)
    # Create the backtest environment
    BtEnv = BackTestEnv(dfs=dfs,
                        dates=rebalance.rebalance_dates,
                        symbols=universe.symbols)
    # Create alpha
    alphas = NeutralizePE(env=BtEnv)
    alphas.operate(BtEnv.match_env(dfs['PE']))
    # run backtest
    bt = QuickBackTesting01(env=BtEnv,
                            universe=universe,
                            n_groups=10)
    bt.run_backtest(alphas.data)

    #
    # from quant.vbt.stats import ic
    # icSeries = ic(alphas.data, BtEnv._FutureReturn, method='spearman')

    # plot the result
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    color_generator = ColorGenerator(10)
    colors = color_generator()
    for i in range(10):
        ax.plot((1+bt.returns.iloc[:, i]).cumprod(), label=f'group_{i+1}', color=colors[i])
    # temp = benchmark.data.loc[rebalance.rebalance_dates]['Close'].pct_change()
    # temp.fillna(0, inplace=True)
    # ax.plot(temp.cumsum(), label='benchmark', color='blue')
    # ax.plot((bt.returns['group_10']).cumsum()-temp.cumsum(), label='excess', color='orange')
    plt.legend(fontsize=16)
    plt.show()


