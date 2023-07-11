from matplotlib import gridspec
import numpy as np
import pandas as pd
from torchqtm.edbt.temp.datahandler import BarData
from torchqtm.edbt.temp.backtest import BacktestEngine
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

lm = LinearRegression()
factor_name = 'fom123_mean'


class FixedPeriodRebalanceBacktester(BacktestEngine):

    def initialize(self):
        return None

    def handle_data(self):
        if self.current_bar < 22:
            print("有内鬼, 停止交易", self.current_bar)
            return None

        if self.bars.now in self.rebalance_date:
            print(self.now)
            account1 = self.get_account('account1')
            account2 = self.get_account('account2')
            num_of_stocks = 300
            status = self.history('status')
            cond_lst = status[status == 0].index  # 排除停牌
            factor_data = self.history('pb')
            float_mv = self.history('float_mv')
            ind_data = self.history('sector1')
            mymap = factor_data.groupby(ind_data).apply(np.mean)
            # 行业中性化
            factor_data = factor_data - ind_data.map(mymap).astype('float64')
            # 市值中性化
            data = pd.concat([factor_data, float_mv], axis=1)
            data.columns = ['factor_data', 'float_mv']
            data = data.dropna()
            data['intercept'] = 1
            model = sm.OLS(data['factor_data'], data[['intercept', 'float_mv']]).fit()
            residuals = data['factor_data'] - model.predict(data[['intercept', 'float_mv']])
            factor_data.loc[data.index] = residuals

            buy_lst = factor_data.nlargest(num_of_stocks).dropna().index
            buy_lst = buy_lst.intersection(cond_lst).to_list()
            buy_lst_2 = factor_data.nsmallest(num_of_stocks).dropna().index
            buy_lst_2 = buy_lst_2.intersection(cond_lst).to_list()
            if not buy_lst:
                return None
            weights = [1 / len(buy_lst)] * len(buy_lst)
            weights2 = [1 / len(buy_lst_2)] * len(buy_lst_2)
            account1.order_pct_to(buy_lst, weights)
            account2.order_pct_to(buy_lst_2, weights2)

        return None


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle
    import copy

    # data = pd.DataFrame(np.random.normal(3000, 300, size=(1000, len(stocks))), columns=stocks,
    #                     index=pd.date_range('2000-01-01', periods=1000))
    # with open('./CICC/project1/datas.pkl', 'rb') as f:
    with open('quotes_dict.pkl', 'rb') as f:
        dfs = pickle.load(f)
    tempdata = copy.deepcopy(dfs['Close'])
    bar = BarData(dfs, start_date='20100101', end_date='20230401', lookback=60)
    # bar = ModifiedBarData(dfs, start_date='20100101', end_date='20230401', lookback=60)
    temp = FixedPeriodRebalanceBacktester(bar, ['account1', 'account2'])
    temp.run_strategy()
    rlt = temp.run_stats()

    fig = plt.figure(figsize=(20, 14), facecolor='white')
    gs = gridspec.GridSpec(4, 2, left=0.04, bottom=0.15, right=0.96, top=0.96, wspace=None, hspace=0,
                           height_ratios=[5, 1, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(range(len(rlt['account1'])), (rlt['account1'] + 1).cumprod().values, color='red')
    ax.plot(range(len(rlt['account1'])), (rlt['account2'] + 1).cumprod().values, color='blue')

    ax.set_xlim(0, len(rlt['account1']))
    step = int(len(rlt['account1']) / 5)
    ax.set_xticks(range(0, len(rlt['account1']), step))
    ax.set_xticklabels(temp.bars.time_index.strftime('%Y-%m-%d')[::step])
    plt.title(f'{factor_name}')
    fig.show()
    # fig.savefig(f'./result/{factor_name}.pdf')
