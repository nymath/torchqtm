from torchqtm._C._rolling import *
import numpy as np
from typing import Callable, Union
from numpy.typing import NDArray
import datetime
import functools

# PythonScalar = Union[str, float, bool]
# DatetimeLikeScalar = Union["Period", "Timestamp", "Timedelta"]
# PandasScalar = Union["Period", "Timestamp", "Timedelta", "Interval"]
# Scalar = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, datetime]


@functools.lru_cache(maxsize=None)
def get_window_bounds(num_values, window_size):
    offset = 0
    end = np.arange(1 + offset, num_values + 1 + offset, 1, dtype=np.int64)
    start = end - window_size
    end = np.clip(end, 0, num_values)
    start = np.clip(start, 0, num_values)
    return start, end


def roll_apply(array: NDArray[np.float64],
               window_size: int,
               func: Callable[..., Union[np.float64, np.array]],
               *args,
               **kwargs) -> NDArray[np.float64]:
    start, end = get_window_bounds(len(array), window_size)
    if array.ndim == 1:
        return roll_apply_1D(array, start, end, window_size, func, args, kwargs)
    elif array.ndim == 2:
        return roll_apply_2D(array, start, end, window_size, func, args, kwargs)
    else:
        raise ValueError("Invalid number of dimensions")


def roll_apply_max(array: NDArray[np.float64],
                   window_size: int) -> NDArray[np.float64]:
    start, end = get_window_bounds(len(array), window_size)
    if array.ndim == 1:
        return roll_max(array, start, end, window_size)
    elif array.ndim == 2:
        if array.shape[1] <= 500:
            rlt = np.empty_like(array)
            for i in range(array.shape[1]):
                rlt[:, i] = roll_max(array[:, i], start, end, window_size)
            return rlt
        else:
            return roll_apply_2D(array, start, end, window_size, lambda x: np.max(x, axis=0), (), {})
    else:
        raise ValueError("Invalid number of dimensions")


def roll_apply_min(array: NDArray[np.float64],
                   window_size: int) -> NDArray[np.float64]:
    start, end = get_window_bounds(len(array), window_size)
    if array.ndim == 1:
        return roll_max(array, start, end, window_size)
    elif array.ndim == 2:
        if array.shape[1] <= 500:  # 这里使用500是个经验法则了吧
            rlt = np.empty_like(array)
            for i in range(array.shape[1]):
                rlt[:, i] = roll_min(array[:, i], start, end, window_size)
            return rlt
        else:
            return roll_apply_2D(array, start, end, window_size, lambda x: np.min(x, axis=0), (), {})
    else:
        raise ValueError("Invalid number of dimensions")

