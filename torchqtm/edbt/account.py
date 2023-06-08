from datahandler import BarData
from queue import Queue
import pandas as pd
import numpy as np


class VirtualAccount(object):
    def __init__(self, name, bars: BarData):
        self.name = name
        self.bars = bars
        self.auto_update = True
        self.fills = Queue()
        self.weights = []

    def order_pct_to(self, symbols, weights) -> None:
        assert len(symbols) == len(weights), "the length of symbols and weights must be equal"
        assert np.abs(np.sum(weights) - 1.0) <= 0.001, "the sum of weights must be 1"
        assert self.fills.empty(), "一条bar上仅能设置一个仓位"
        target_weights = self.bars.get_template_cseries
        target_weights.loc[symbols] = weights
        target_weights.fillna(0, inplace=True)
        self.fills.put(target_weights)  # 往订单中放入目标权重
        self.auto_update = False

    def _reset_account(self):  # 系统内部每天调用一次
        self.auto_update = True

    def _passive_update(self):
        if not self.auto_update:
            return None
        if len(self.weights) == 0:
            self.fills.put(self.bars.get_template_cseries.fillna(0))
        else:
            weights_rebalanced = self.weights[-1].copy(deep=True)
            weights_rebalanced = weights_rebalanced * (1 + self.bars.history('PctChange'))
            if np.sum(weights_rebalanced) != 1 and np.sum(weights_rebalanced) != 0:
                weights_rebalanced = weights_rebalanced / np.sum(weights_rebalanced)
            self.fills.put(weights_rebalanced)
        self.auto_update = False

    def _update_from_fills(self):
        assert not self.fills.empty(), "the fills queue is empty"
        self.weights.append(self.fills.get())

    def update_account(self):
        self._passive_update()
        self._update_from_fills()
        self._reset_account()

    def calculate_returns(self):
        weights = pd.concat(self.weights, axis=1).T
        weights.index = self.bars.time_index
        weights = weights.shift(1).fillna(0)
        returns = weights * self.bars.PctChange
        returns = returns.sum(axis=1)
        return returns

