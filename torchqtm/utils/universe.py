import pandas as pd
import tushare as ts
import numpy as np
from torchqtm.config import __TS_API__
from torchqtm.utils import relativedelta, datetime
pro = ts.pro_api(__TS_API__)
from typing import Iterable, Union, List
from abc import abstractmethod, ABCMeta


class Universe(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_symbols(self, trade_date: Union[str, datetime, pd.Timestamp]):
        raise NotImplementedError


class StaticUniverse(Universe):
    def __init__(self, symbols: Iterable[str]):
        super().__init__()
        self.symbols = symbols

    def get_symbols(self, trade_date=None):
        return self.symbols

    @property
    def data(self):
        return self.symbols


class DynamicUniverse(object):
    pass


class IndexComponents(object):
    def __init__(self,
                 index_code: str,
                 in_date: str):
        self.index_code = index_code
        self.in_date = in_date
        self.data = self._get_components()

    def _get_components(self) -> List[str]:
        end = datetime.strptime(self.in_date, "%Y%m%d")
        start = end - relativedelta(months=1)
        df = pro.index_weight(index_code='399300.SZ', start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        rlt = df[df['trade_date'] == max(np.unique(df['trade_date']))]['con_code']
        rlt = sorted(list(rlt.apply(lambda x: x[:-3])))
        return rlt
