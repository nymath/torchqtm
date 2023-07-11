import typing
import builtins
import pandas as pd
import datetime
from enum import Enum

# from builtins import FuncType
Callable = typing.Callable
Any = typing.Any
TypeVar = typing.TypeVar

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
DocType = Callable[[FuncType], FuncType]  # 装饰器空间
D = TypeVar("D", bound=DocType)

# Callable[[...], DocType], 这个就是pandas中那个doc属于的空间

List = typing.List
Tuple = typing.Tuple
Dict = typing.Dict
Union = typing.Union
Optional = typing.Optional
Iterator = typing.Iterator
Sequence = typing.Sequence
Iterable = typing.Iterable
Literal = typing.Literal

BASE_FIELDS = typing.Literal['open', 'high', 'low', 'close', 'volume', 'price', 'contract', 'last_traded']
OHLCV_FIELDS = typing.Literal['open', 'high', 'low', 'close', 'volume']
OHLCVP_FIELDS = typing.Literal['open', 'high', 'low', 'close', 'volume', 'price']

# 这俩实在是太容易搞混了
HISTORY_FREQUENCIES = typing.Literal['1m', '1d']
DATA_FREQUENCIES = typing.Literal["minute", "daily"]


DateLike = typing.Union[pd.Timestamp, str, int, float, datetime.datetime]


class ORDER_STATUS(Enum):
    OPEN = 0
    FILLED = 1
    CANCELLED = 2
    REJECTED = 3
    HELD = 4


class ORDER_TYPE(Enum):
    MARKET = 0
    LIMIT = 1
    STOP = 2
    STOP_LIMIT = 3


class EVENT_TYPE(Enum):
    BAR = 0  # end of bar
    SESSION_START = 1  # start of a trading day
    SESSION_END = 2  # end of a trading day
    MINUTE_START = 3
    MINUTE_END = 4
    BEFORE_TRADING_START = 5  # start of trading
    MARKET_OPEN = 6


class ASSET_TYPE(Enum):
    Equity = 0
    Future = 1
    Index = 2
    Fund = 3


class TestError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self, message)






