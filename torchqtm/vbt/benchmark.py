import tushare as ts
import pandas as pd
import numpy as np
from torchqtm.config import __TS_API__
pro = ts.pro_api(__TS_API__)


class BenchMark(object):
    def __init__(self, index_code, start_date, end_date):
        """
        index_code: '000905.SH', '399300.SZ'
        """
        self.index_code = index_code
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._get_index_data()

    def _get_index_data(self):
        df = pro.index_daily(ts_code=self.index_code, start_date=self.start_date, end_date=self.end_date)
        df.sort_values('trade_date', inplace=True)
        df = df.set_index('trade_date')
        df.index = pd.to_datetime(df.index)
        del df['ts_code']
        df.columns = ['Close', 'Open', 'High', 'Low', 'PreClose', 'Change', 'PctChange', 'Volume', 'Amount']
        df['PctChange'] = df['PctChange'] / 100
        return df


if __name__ == '__main__':
    benchmark = BenchMark('000905.SH', '20180101', '20230101')