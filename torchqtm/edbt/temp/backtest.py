from torchqtm.edbt.temp.account import VirtualAccount
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from torchqtm.edbt.temp.datahandler import BarData
from mycalendar import Calendar
import pandas as pd


class BacktestEngine(object, metaclass=ABCMeta):

    def __init__(self, bars: BarData, accounts_names: Iterable[str]):
        self.bars = bars  # 想继承bars的一些方法
        self.accounts = {key: VirtualAccount(key, self.bars) for key in accounts_names}
        self.history = self.bars.history;
        self.calendar = Calendar(start_date=self.bars.start_date, end_date=self.bars.end_date)
        self.rebalance_date = self.create_rebalance_date()

    @property
    def now(self) -> pd.Timestamp:
        return self.bars.now

    @property
    def current_bar(self) -> int:
        return self.bars.bar_index

    def get_account(self, account_name):
        return self.accounts[account_name]

    def create_rebalance_date(self) -> list[pd.Timestamp]:
        temp = self.calendar.create_monthly_groups()  # TODO: 加入参数
        rlt = []
        for x in temp.values():
            rlt.append(x[-1])
        return rlt

    @abstractmethod
    def initialize(self):
        """
        """
        raise NotImplementedError('Should be implemented in subclass')

    @abstractmethod
    def handle_data(self):
        """
        """
        raise NotImplementedError('Should be implemented in subclass')

    # @abstractmethod
    # def generate_alpha_factor(self):
    #     """
    #     """
    #     raise NotImplementedError('Should be implemented in subclass')

    def run_strategy(self):
        self.initialize()

        while True:
            self.bars.update_bars()
            if not self.bars.continue_backtest:  # 没有取出数据的话则停止回测
                break
            self.handle_data()
            for account in self.accounts:
                self.accounts[account].update_account()

    def run_stats(self) -> dict:
        # TODO: 新增一个类, 用来计算各种指标
        rlt = {}
        for key in self.accounts:
            rlt[key] = self.accounts[key].calculate_returns()
        return rlt


class QuickBacktestEngine(object, metaclass=ABCMeta):

    pass