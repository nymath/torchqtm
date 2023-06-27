from zipline.api import order, record, symbol
from zipline.finance import commission, slippage
from zipline import run_algorithm
from zipline.algorithm import TradingAlgorithm
import pandas as pd
from zipline._protocol import BarData


def initialize(context: TradingAlgorithm):
    context.sym = [symbol('AAPL'), symbol('AMD')]
    context.i = 0

    context.set_commission(commission.PerShare(cost=.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())


def handle_data(context: TradingAlgorithm, data: BarData):
    # Skip first 300 days to get full windows
    context.i += 1
    # 首次进入的话是当天晚上收盘的时候
    data.history(context.sym, ['open', 'close', 'high', 'low'], 4, '1d')
    if context.i == 1:
        print("Now we can buy APPLE")
        context.order_target(context.sym[0], 1000, limit_price=100)
        context.order_target(context.sym[1], 1000, limit_price=100)

    print(context.portfolio)
    # # Compute averages
    # if short_mavg > long_mavg:
    #     # order_target orders as many shares as needed to
    #     # achieve the desired number of shares.
    #     for i in range(3):
    #         context.order_target(context.sym, 1)
    # elif short_mavg < long_mavg:
    #     context.order_target(context.sym, 0)
    #
    # # Save values for later inspection
    # record(AAPL=data.current(context.sym, "price"),
    #        short_mavg=short_mavg,
    #        long_mavg=long_mavg)


# def analyze(context=None, results=None):
#     import matplotlib.pyplot as plt
#     # Plot the portfolio and asset data.
#     ax1 = plt.subplot(211)
#     results.portfolio_value.plot(ax=ax1)
#     ax1.set_ylabel('Portfolio value (USD)')
#     # ax2 = plt.subplot(212, sharex=ax1)
#     # results.AAPL.plot(ax=ax2)
#     # ax2.set_ylabel('AAPL price (USD)')
#
#     # Show the plot.
#     plt.gcf().set_size_inches(18, 8)
#     plt.show()


if __name__ == '__main__':
    capital_base = 200000
    start = pd.to_datetime('2015-01-02')
    end = pd.to_datetime('2018-01-01')
    result = run_algorithm(start=start, end=end, initialize=initialize,
                           capital_base=capital_base, handle_data=handle_data,
                           bundle='quandl')

