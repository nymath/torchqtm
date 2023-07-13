from torchqtm._C._rolling import *
from pandas._libs.window.aggregations import (
    roll_rank,
    roll_quantile,
    roll_kurt,
    roll_var,
    roll_weighted_sum
)
from pandas._libs.algos import rank_1d, rank_2d
import numpy as np
from typing import Callable, Union
from numpy.typing import NDArray
import datetime
import functools


# PythonScalar = Union[str, float, bool]
# DatetimeLikeScalar = Union["Period", "Timestamp", "Timedelta"]
# PandasScalar = Union["Period", "Timestamp", "Timedelta", "Interval"]
# Scalar = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, datetime]


# @functools.lru_cache(maxsize=None)
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
    elif array.ndim == 3:
        return roll_apply_3D(array, start, end, window_size, func, args, kwargs)
    else:
        raise ValueError("Invalid number of dimensions")


def roll_apply_max(array: NDArray[np.float64],
                   window_size: int) -> NDArray[np.float64]:
    start, end = get_window_bounds(len(array), window_size)
    if array.ndim == 1:
        return roll_max(array, start, end, window_size)
    elif array.ndim == 2:
        if array.shape[1] <= 500 or window_size > 100:
            rlt = np.empty_like(array)
            for i in range(array.shape[1]):
                rlt[:, i] = roll_max(array[:, i], start, end, window_size)
            return rlt
        else:
            return roll_apply_2D(array, start, end, window_size, lambda x: np.nanmax(x, axis=0), (), {})
    else:
        raise ValueError("Invalid number of dimensions")


def roll_apply_min(array: NDArray[np.float64],
                   window_size: int) -> NDArray[np.float64]:
    start, end = get_window_bounds(len(array), window_size)
    if array.ndim == 1:
        return roll_min(array, start, end, window_size)
    elif array.ndim == 2:
        if array.shape[1] <= 500 or window_size > 100:  # 这里使用500是个经验法则了吧
            rlt = np.empty_like(array)
            for i in range(array.shape[1]):
                rlt[:, i] = roll_min(array[:, i], start, end, window_size)
            return rlt
        else:
            return roll_apply_2D(array, start, end, window_size, lambda x: np.nanmin(x, axis=0), (), {})
    else:
        raise ValueError("Invalid number of dimensions")


def roll_apply_mean(array: NDArray[np.float64],
                    window_size: int,
                    mode:str = "auto") -> NDArray[np.float64]:
    start, end = get_window_bounds(len(array), window_size)
    aux_func = roll_mean
    helper_func = lambda x: np.nanmean(x, axis=0)

    if array.ndim == 1:
        return aux_func(array, start, end, window_size)
    if mode == "auto":
        if array.ndim == 2:
            if window_size > 20 or array.shape[1] < 20:  # 这里使用500是个经验法则了吧
                rlt = np.empty_like(array)
                for i in range(array.shape[1]):
                    rlt[:, i] = aux_func(array[:, i], start, end, window_size)
                return rlt
            else:
                return roll_apply_2D(array, start, end, window_size, helper_func, (), {})
        else:
            raise ValueError("Invalid number of dimensions")

    elif mode == "overall":
        return roll_apply_2D(array, start, end, window_size, helper_func, (), {})

    elif mode == "single":
        if array.ndim == 2:
            rlt = np.empty_like(array)
            for i in range(array.shape[1]):
                rlt[:, i] = aux_func(array[:, i], start, end, window_size)
            return rlt


def roll_apply_sum(array: NDArray[np.float64],
                   window_size: int,
                   mode: str = "auto") -> NDArray[np.float64]:
    start, end = get_window_bounds(len(array), window_size)
    aux_func = roll_sum
    helper_func = lambda x: np.nansum(x, axis=0)

    if array.ndim == 1:
        return aux_func(array)
    if mode == "auto":
        if array.ndim == 2:
            if window_size > 50:  # 这里使用500是个经验法则了吧
                rlt = np.empty_like(array)
                for i in range(array.shape[1]):
                    rlt[:, i] = aux_func(array[:, i], start, end, window_size)
                return rlt
            else:
                return roll_apply_2D(array, start, end, window_size, helper_func, (), {})
        else:
            raise ValueError("Invalid number of dimensions")

    elif mode == "overall":
        return roll_apply_2D(array, start, end, window_size, helper_func, (), {})

    elif mode == "single":
        if array.ndim == 2:
            rlt = np.empty_like(array)
            for i in range(array.shape[1]):
                rlt[:, i] = aux_func(array[:, i], start, end, window_size)
            return rlt


def roll_apply_rank(array: NDArray[np.float64],
                    window_size: int,
                    method="min",
                    percentile: bool = False,
                    ascending: bool = True,
                    mode: str = "auto") -> NDArray[np.float64]:
    start, end = get_window_bounds(len(array), window_size)
    aux_func = roll_rank
    helper_func = lambda x: rank_2d(x[::-1], axis=0)[-1]
    if array.ndim == 1:
        return aux_func(array, start, end, window_size, percentile, method, ascending)
    if mode == "auto":
        if array.ndim == 2:
            if window_size > 20:
                rlt = np.empty_like(array)
                for i in range(array.shape[1]):
                    rlt[:, i] = aux_func(array[:, i], start, end, window_size, percentile, method, ascending)
                return rlt
            else:
                return roll_apply_2D(array, start, end, window_size, helper_func, (), {})
        else:
            raise ValueError("Invalid number of dimensions")

    elif mode == "overall":
        return roll_apply_2D(array, start, end, window_size, helper_func, (), {})

    elif mode == "single":
        if array.ndim == 2:
            rlt = np.empty_like(array)
            for i in range(array.shape[1]):
                rlt[:, i] = aux_func(array[:, i], start, end, window_size, percentile, method, ascending)
            return rlt


