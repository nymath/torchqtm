import copy

import numpy as np
import pandas as pd
from collections import deque
from typing import Union
import multiprocessing as mp
import concurrent.futures
from joblib import Parallel, delayed
import os
import sys
import time
import pickle

# TODO: 增加数据源


class BarData(object):
    def __init__(self, datas, start_date, end_date, lookback: int):
        """
        如果没有OHLCV数据, 那么请用0进行填充
        :param datas:
        """
        # assert 'Open' in datas, 'Open is required'
        # assert 'High' in datas, 'High is required'
        # assert 'Low' in datas, 'Low is required'
        # assert 'Close' in datas, 'Close is required'
        # assert 'Volume' in datas, 'Volume is required'
        assert isinstance(lookback, int)
        self.datas = datas
        self.start_date = start_date
        self.end_date = end_date
        if start_date is not None and end_date is not None:
            for key in self.datas:
                self.datas[key] = self.datas[key].loc[start_date:end_date]
        # TODO: 这个pctchange衡量了交易的价格
        self.datas['PctChange'] = self.datas['Close'].pct_change().fillna(0)
        self.PctChange = ...
        self.__create_PctChange()
        self.lookback = lookback
        self.time_index = self.datas['Close'].index
        self.attrs = list(self.datas.keys())
        self.stocks = self.datas['Close'].columns

        self.continue_backtest = True  # 也可以看作, 当天是否取出了数据
        self.bar_index = -1
        self.time_buffers, self.data_buffers = self._create_buffers()
        # self.__save_structure_data()
        self._convert_datas()
        print('success')

    def __create_PctChange(self) -> None:
        self.PctChange = self.datas['Close'].pct_change().fillna(0)

    def __save_structure_data(self):
        with open('modified_datas.pkl', 'wb') as f:
            temp = copy.deepcopy(self.datas)
            for key in self.datas:
                temp[key] = list(self.datas[key].iterrows())
            pickle.dump(temp, f)

    def _create_buffers(self):
        return deque(maxlen=self.lookback), {key: deque(maxlen=self.lookback) for key in self.datas.keys()}

    def _convert_datas(self):
        from queue import Queue
        for key in self.datas:
            self.datas[key] = self.datas[key].iterrows()

    def update_bars(self) -> None:
        """
        Pushes the latest bar to the history_symbol_data structure
        for all symbols in the self.symbol_list
        """

        self.bar_index += 1
        try:
            for key in self.datas:
                current_bar = next(self.datas[key])
                if key == 'Close':
                    self.time_buffers.append(current_bar[0])
                else:
                    self.data_buffers[key].append(current_bar[1])

        except StopIteration:
            self.continue_backtest = False

    def history(self, attr, N=1) -> Union[pd.DataFrame, pd.Series]:
        if N == 1:
            return self.data_buffers[attr][-1]
        else:
            return pd.concat(list(self.data_buffers[attr])[-N:], axis=1).T

    @property
    def now(self):
        return self.time_buffers[-1]

    @property
    def get_template_frame(self):
        return pd.DataFrame(np.nan, index=self.time_index, columns=self.stocks)

    @property
    def get_template_tseries(self):
        return pd.Series(np.nan, index=self.time_index)

    @property
    def get_template_cseries(self):
        return pd.Series(np.nan, index=self.stocks)


def update_bars_process(data, key, time_buffers, data_buffers):
    try:
        current_bar = next(data)
        if key == 'Close':
            time_buffers.put(current_bar[0])
        else:
            data_buffers[key].put(current_bar[1])

    except StopIteration:
        pass


class MultiBarData(BarData):
    def __convert_dataframe(self, key):
        return self.datas[key].iterrows()

    def update_bars(self) -> None:
        """
        Pushes the latest bar to the history_symbol_data structure
        for all symbols in the self.symbol_list
        """

        self.bar_index += 1

        processes = []
        for key in self.datas:
            process = mp.Process(target=update_bars_process, args=(self.datas[key], key, self.time_buffers, self.data_buffers))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        try:
            for key in self.data_buffers:
                while not self.data_buffers[key].empty():
                    self.data_buffers_list[key].append(self.data_buffers[key].get())

        except StopIteration:
            self.continue_backtest = False


def update_bars_process(data, key, time_buffers, data_buffers):
    try:
        current_bar = next(data)
        if key == 'Close':
            time_buffers.put(current_bar[0])
        else:
            data_buffers[key].put(current_bar[1])

    except StopIteration:
        pass


class ModifiedBarData(BarData):
    def __init__(self, datas, start_date, end_date, lookback: int):
        """
        如果没有OHLCV数据, 那么请用0进行填充
        :param datas:
        """
        # assert 'Open' in datas, 'Open is required'
        # assert 'High' in datas, 'High is required'
        # assert 'Low' in datas, 'Low is required'
        # assert 'Close' in datas, 'Close is required'
        # assert 'Volume' in datas, 'Volume is required'
        assert isinstance(lookback, int)
        self.datas = datas
        self.start_date = start_date
        self.end_date = end_date
        # TODO: 这个pctchange衡量了交易的价格
        self.PctChange = ...
        self.__create_PctChange()
        self.lookback = lookback
        self.time_index = pd.DatetimeIndex([x[0] for x in self.datas['Close']])
        self.attrs = list(self.datas.keys())

        self.continue_backtest = True  # 也可以看作, 当天是否取出了数据
        self.bar_index = -1
        self.time_buffers, self.data_buffers = self._create_buffers()
        self.stocks = self.datas['Close'][0][1].index
        # self.__save_structure_data()
        self._convert_datas()
        print('success')

    def __create_PctChange(self) -> None:
        self.PctChange = pd.concat([x[1] for x in self.datas['Close']], axis=1).T.pct_change().fillna(0)

    def _convert_datas(self):
        return None

    def update_bars(self) -> None:
        """
        Pushes the latest bar to the history_symbol_data structure
        for all symbols in the self.symbol_list
        """

        self.bar_index += 1
        try:
            for key in self.datas:
                current_bar = self.datas[key].pop(0)
                if key == 'Close':
                    self.time_buffers.append(current_bar[0])
                else:
                    self.data_buffers[key].append(current_bar[1])

        except IndexError:
            self.continue_backtest = False
