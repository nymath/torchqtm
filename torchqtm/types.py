from typing import TypeVar
import builtins
# from builtins import FuncType
from typing import TypeVar, Callable, Any
from pandas.util._decorators import doc
import torch.nn.functional
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
DocType = Callable[[FuncType], FuncType]
D = TypeVar("D", bound=DocType)
# Callable[[F], F], 这个是装饰器空间
# Callable[[...], DocType], 这个就是pandas中那个doc属于的空间
