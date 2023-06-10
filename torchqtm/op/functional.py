import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..config import __OP_MODE__
from typing import overload
from .algos import rank_1d, rank_2d
from scipy.stats import norm
import talib

# Arithmetic Operators


def abs(x):
    """absolute value of x"""
    if isinstance(x, pd.DataFrame):
        return np.abs(x)
    elif isinstance(x, np.ndarray):
        return np.abs(x)


def ceiling(X):
    """ceiling of x"""
    return np.ceil(X)


def floor(X):
    return np.floor(X)


def divide(X, Y):
    return np.divide(X, Y)


# TODO: Revise it

def add(X, Y):
    return np.add(X, Y)


def sub(X, Y):
    return np.subtract(X, Y)


def mul(X, Y):
    np.multiply(X, Y)


def pow(X, p):
    return np.power(X, p)


def less(X, Y):
    return np.less(X, Y)


def more(X, Y):
    return np.greater(X, Y)


def leq(X, Y):
    return np.less_equal(X, Y)


def geq(X, Y):
    return np.greater_equal(X, Y)


def eq(X, Y):
    return np.equal(X, Y)


# def exp(X):
#     if isinstance(X, np.ndarray):
#         return np.exp(X)
#     elif isinstance(X, pd.DataFrame):
#         return X.apply(np.exp, axis=1)


def inverse(X):
    return divide(1, X)


def log(X):
    return np.log(X)


def log_diff(X):
    return np.log(X / ts_delay(X, 1))


def nan_out(X, lower=-0.1, upper=0.1):
    """set returns outside of [lower, upper] to nan"""
    rlt = np.where(((X < lower) + (X > upper)) > 0, np.nan, X)
    if isinstance(rlt, np.ndarray):
        return rlt
    elif isinstance(rlt, pd.DataFrame):
        return pd.DataFrame(rlt, index=X.index, columns=X.columns)


def purify(X):
    rlt = np.where(np.isinf(X), np.nan, X)
    if isinstance(rlt, np.ndarray):
        return rlt
    elif isinstance(rlt, pd.DataFrame):
        return pd.DataFrame(rlt, index=X.index, columns=X.columns)


def sign(X):
    rlt = np.where(X > 0, 1, X)
    rlt = np.where(X < 0, -1, rlt)
    rlt = np.where(X == 0, 0, rlt)
    if isinstance(rlt, np.ndarray):
        return rlt
    elif isinstance(rlt, pd.DataFrame):
        return pd.DataFrame(rlt, index=X.index, columns=X.columns)


def sqrt(x):
    return np.sqrt(x)


def to_nan(X, value=0, reverse=False):
    if not reverse:
        rlt = np.where(X == value, np.nan, X)
    else:
        rlt = np.where(X != value, np.nan, X)
    if isinstance(rlt, np.ndarray):
        return rlt
    elif isinstance(rlt, pd.DataFrame):
        return pd.DataFrame(rlt, index=X.index, columns=X.columns)


def densify(X):
    # TODO: implement this
    pass


# Logical Operators
# TODO: 1231
def if_else(condition, value_if_true, value_if_false):
    rlt = np.where(condition, value_if_true, value_if_false)
    if isinstance(rlt, np.ndarray):
        return rlt
    elif isinstance(rlt, pd.DataFrame):
        return pd.DataFrame(rlt, index=condition.index, columns=condition.columns)


def logical_and(x, y):
    return np.logical_and(x, y)


def logical_or(x, y):
    return np.logical_or(x, y)


def negate(x):
    return np.negative(x)


def is_nan(X):
    return np.isnan(X)


def if_finite(X):
    rlt = np.where(np.isfinite(X), 1, 0)
    if isinstance(rlt, np.ndarray):
        return rlt
    elif isinstance(rlt, pd.DataFrame):
        return pd.DataFrame(rlt, index=X.index, columns=X.columns)


# Time Series Operators
from torchqtm.core.window.rolling import (
    roll_apply,
    roll_apply_max,
    roll_apply_min,
    roll_apply_mean,
    roll_apply_sum,
    roll_apply_rank,
)


def ts_apply(x, d, func):
    if isinstance(x, np.ndarray):
        return roll_apply(x, d, func)
    elif isinstance(x, pd.Series):
        return pd.Series(roll_apply(x, d, func), index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(roll_apply(x.values, d, func), index=x.index, columns=x.columns)


def ts_mean(x, d):
    def aux_func(array, window_size):
        return roll_apply_mean(array, window_size)

    if isinstance(x, np.ndarray):
        return aux_func(x, d)
    elif isinstance(x, pd.Series):
        return pd.Series(aux_func(x.values, d), index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(aux_func(x.values, d), index=x.index, columns=x.columns)


def ts_max(x, d):
    def aux_fun(array, window_size):
        return roll_apply_max(array, window_size)

    if isinstance(x, np.ndarray):
        return aux_fun(x, d)
    elif isinstance(x, pd.Series):
        return pd.Series(aux_fun(x.values, d), index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(aux_fun(x.values, d), index=x.index, columns=x.columns)


def ts_min(x, d):
    def aux_func(array, window_size):
        return roll_apply_min(array, window_size)

    if isinstance(x, np.ndarray):
        return aux_func(x, d)
    elif isinstance(x, pd.Series):
        return pd.Series(aux_func(x.values, d), index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(aux_func(x.values, d), index=x.index, columns=x.columns)


def ts_sum(x, d, mode='auto'):
    def aux_func(array, window_size):
        return roll_apply_sum(array, window_size, mode=mode)

    if isinstance(x, np.ndarray):
        return aux_func(x, d)
    elif isinstance(x, pd.Series):
        return pd.Series(aux_func(x.values, d), index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(aux_func(x.values, d), index=x.index, columns=x.columns)


def ts_delay(x, d):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.shift(d)
    elif isinstance(x, np.ndarray):
        return pd.DataFrame(x).shift(d).values
    else:
        raise TypeError("Input should be a pandas DataFrame, Series or a numpy ndarray.")


def ts_delta(x, d):
    return sub(x, ts_delay(x, d))


def days_from_last_change(x, d):
    pass


def ts_weighted_delay(x, k=0.5):
    return add(k * x, (1-k) * ts_delay(x, 1))


# 这俩函数有点抽象, 后边想想怎么实现
# def hump(x, hump=0.01):
#     """
#     This operator limits amount and magnitude of changes in input
#     (thus reducing turnover). If input changed by less than a threshold
#      compared to previous days’ output, current output is the same as on the previous day.
#     """
#
#
#
# def hump_decay(x, p=0):
#     pass


def inst_tvr(x, d):
    """
    Total trading value / Total holding value in the past d days

    """
    pass


def jump_decay(x, d, sensitivity=0.5, force=0.1):
    """
    If there is a huge jump in current data compare to previous one, apply force:
    我认为作用是滤平变动
    """
    cond = abs(x-ts_delay(x, 1) > sensitivity * ts_std_dev(x, d))
    value_if_true = ts_delay(x, 1) + ts_delta(x, 1) * force
    value_if_false = x
    return if_else(cond, value_if_true, value_if_false)


def kth_element(x, d, k):
    pass


def last_diff_value(x, d):
    pass


def ts_arg_min(x, d):
    def aux_func(t):
        return np.nanargmin(t[::-1], axis=0)

    return ts_apply(x, d, aux_func)


def ts_arg_max(x, d):
    def aux_func(t):
        return np.nanargmax(t[::-1], axis=0)

    return ts_apply(x, d, aux_func)


def ts_av_diff(x, d):
    return sub(x, ts_mean(x, d))


def ts_backfill(x, d, k=1, ignore="NAN"):
    pass


def ts_co_kurtosis(y, x, d):
    pass


def _ts_corr_single(x, y, d):
    pass


def ts_corr(x, y, d):
    """
    Pearson correlation of x, y in the past d days.
    """
    if len(x.shape) == 2:
        rlt = pd.DataFrame(x).rolling(d).corr(pd.DataFrame(y))
    else:
        rlt = pd.Series(x).rolling(d).corr(pd.Series(y))
    if isinstance(x, np.ndarray):
        return rlt.values
    elif isinstance(x, pd.Series):
        return pd.Series(rlt.values, index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(rlt.values, index=x.index, columns=x.columns)


def ts_co_skewness(y, x, d):
    pass


def ts_count_nans(x, d):
    def aux_func(t):
        return np.nansum(np.isnan(t), axis=0)

    return ts_apply(x, d, aux_func)


def ts_covariance(y, x, d):
    pass


def ts_decay_exp_window(x, d, factor):
    pass


def ts_decay_linear(x, d, dense=False):
    # TODO: 想想怎么改进这个矩阵乘法, 目前会产生大量的NA
    def aux_func(data_slice):
        filters = np.arange(1, d+1) / ( d * (d + 2) / 2 )
        return np.matmul(data_slice.T, filters)
    return ts_apply(x, d, aux_func)


def ts_std_dev(x, d):
    def aux_func(t):
        return np.nanstd(t, axis=0)

    return ts_apply(x, d, aux_func)


def ts_ir(x, d):
    return divide(ts_mean(x, d), ts_std_dev(x, d))


def ts_kurtosis(x, d):
    pass


def ts_max_diff(x, d):
    return sub(x, ts_max(x, d))


def ts_median(x, d):
    def aux_func(t):
        return np.nanmedian(x, axis=0)
    return ts_apply(x, d, aux_func)


def ts_min_diff(x, d):
    return x - ts_min(x, d)


def ts_min_max_cps(x, d, f=2):
    return sub(add(ts_max(x, d), ts_min(x, d)), f * x)


def ts_min_max_diff(x, d, f=0.5):
    return sub(x, f * (ts_max(x, d) + ts_min(x, d)))


def ts_moment(x, d, k=0):
    def aux_func(t):
        return np.nanmean(np.power(t, k), axis=0)

    return ts_apply(x, d, aux_func)


def ts_partial_corr(x, y, z, d):
    pass


def ts_percentage(x, d, percentage=0.5):
    pass


def ts_poly_regression(y, x, d, k=1):
    pass


def ts_product(x, d):
    def aux_func(t):
        return np.nanprod(x, axis=0)

    return ts_apply(x, d, aux_func)


def ts_rank(x, d, mode="auto"):
    def aux_func(array, window_size):
        return roll_apply_rank(array, window_size, method=mode)

    if isinstance(x, np.ndarray):
        return aux_func(x, d)
    elif isinstance(x, pd.Series):
        return pd.Series(aux_func(x.values, d), index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(aux_func(x.values, d), index=x.index, columns=x.columns)


def ts_regression(y, x, d, lag=0, rettype=0):
    pass


def ts_returns(x, d):
    return (x - ts_delay(x, d)) / ts_delay(x, d)


def ts_scale(x, d, constant=0):
    return constant + divide(x - ts_min(x, d), ts_max(x, d) - ts_min(x, d))


def ts_skewness(x, d):
    pass


# ts_step(1), step(1)


def ts_theilsen(x, y, d):
    pass


def ts_triple_corr(x, y, z, d):
    pass


def ts_zscore(x, d):
    return divide(x - ts_mean(x, d), ts_std_dev(x, d))


def ts_entropy(x, d):
    pass


def ts_vector_neut(x, y, d):
    pass


def ts_vector_proj(x, y, d):
    pass


# ts_rank_gmean_amean_diff(input1, input2, input3,...,d)


def _ts_quantile_uniform(x, d):
    def aux_func(array, window_size):
        return ts_rank(array, window_size) / window_size

    if isinstance(x, np.ndarray):
        return aux_func(x, d)
    elif isinstance(x, pd.Series):
        return pd.Series(aux_func(x.values, d), index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(aux_func(x.values, d), index=x.index, columns=x.columns)


def _ts_quantile_gaussian(x, d):
    def aux_fun(array, window_size):
        return norm.ppf(ts_rank(array, window_size) / window_size)

    if isinstance(x, np.ndarray):
        return aux_fun(x, d)
    elif isinstance(x, pd.Series):
        return pd.Series(aux_fun(x.values, d), index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(aux_fun(x.values, d), index=x.index, columns=x.columns)


def ts_quantile(x, d, driver="uniform"):
    if driver == "uniform":
        return _ts_quantile_uniform(x, d)
    elif driver == "gaussian":
        return _ts_quantile_gaussian(x, d)
    else:
        return ValueError


# CrossSectionalOperators
def _cs_corr(X, Y):
    mean_x = np.nanmean(X, axis=1, keepdims=True)
    std_x = np.nanstd(X, axis=1, keepdims=True)
    mean_y = np.nanmean(Y, axis=1, keepdims=True)
    std_y = np.nanstd(Y, axis=1, keepdims=True)
    return np.nanmean((X - mean_x) / std_x * (Y - mean_y) / std_y, axis=1)


def cs_corr(X, Y, method="pearson"):
    if method == "pearson":
        rlt = _cs_corr(X, Y)
    elif method == "spearman":
        # temp_X = pd.DataFrame(X).rank(axis=1)
        # temp_Y = pd.DataFrame(Y).rank(axis=1)
        temp_X = rank_2d(X, axis=1)
        temp_Y = rank_2d(Y, axis=1)
        temp_X = temp_X / np.nanmax(temp_X, axis=1, keepdims=True)
        temp_Y = temp_Y / np.nanmax(temp_Y, axis=1, keepdims=True)
        rlt = _cs_corr(temp_X, temp_Y)
    else:
        raise ValueError("the method should be pearson or spearman")
    if isinstance(X, np.ndarray):
        return rlt
    elif isinstance(X, pd.DataFrame):
        return pd.Series(rlt, index=X.index, name=f'{method}')


def cs_rank(X):
    assert len(X.shape) == 2

    def aux_func(t):
        return rank_2d(t, axis=1)

    if isinstance(X, np.ndarray):
        return aux_func(X)
    elif isinstance(X, pd.DataFrame):
        return pd.DataFrame(aux_func(X.values), index=X.index, columns=X.columns)


def _regression_neut(Y, X):
    base_shape = Y.shape
    n_axis = len(Y.shape)
    data = np.concatenate([Y.reshape(*base_shape, 1), X.reshape(*base_shape, 1)], axis=n_axis)

    def aux_func(data_slice):
        y = data_slice[..., 0]
        x = data_slice[..., 1]
        y_demean = y - np.nanmean(y)
        x_demean = x - np.nanmean(x)
        residuals = y_demean - (np.nanmean(y_demean * x_demean) / np.nanvar(x_demean)) * x_demean
        return residuals

    return ts_apply(data, 1, aux_func)


def _regression_neuts(Y, others):
    pass
    # result = np.empty_like(Y)
    # for i in range(len(Y)):
    #     y = Y[i]
    #     X = np.concatenate([x[i].reshape(-1, 1) for x in others], axis=1)
    #     model = LinearRegression()
    #     model.fit(X, y)
    #     y_pred = model.predict(X)
    #     residuals = y - y_pred
    #     result[i] = residuals
    # return pd.DataFrame(np.array(result), index=Y.index, columns=Y.columns)


def regression_neut(Y, others):
    if not isinstance(others, list):
        if isinstance(Y, np.ndarray):
            return _regression_neut(Y, others)
        elif isinstance(Y, pd.Series):
            return pd.Series(_regression_neut(Y.values, others.values), index=Y.index, name=Y.name)
        elif isinstance(Y, pd.DataFrame):
            return pd.DataFrame(_regression_neut(Y.values, others.values), index=Y.index, columns=Y.columns)
    else:
        raise ValueError


def regression_proj(Y, others):
    return Y - regression_neut(Y, others)


def _winsorize(X, method, param):
    assert isinstance(X, np.ndarray)
    n_stocks = X.shape[-1]
    n_axis = len(X.shape)
    if method == 'quantile':
        lower_bound = np.nanquantile(X, param, axis=n_axis-1, keepdims=True)
        upper_bound = np.nanquantile(X, 1-param, axis=n_axis-1, keepdims=True)
    elif method == 'std':
        mean = np.nanmean(X, axis=n_axis-1, keepdims=True)
        std = np.nanstd(X, axis=n_axis-1, keepdims=True)
        lower_bound = mean - std * param
        upper_bound = mean + std * param
    else:
        raise ValueError('Unknown method')
    return np.clip(a=X,
                   a_min=np.repeat(lower_bound, n_stocks, axis=n_axis-1),
                   a_max=np.repeat(upper_bound, n_stocks, axis=n_axis-1))


def winsorize(X, method='quantile', param=0.05):
    aux_func = _winsorize
    if isinstance(X, np.ndarray):
        return aux_func(X, method, param)
    elif isinstance(X, pd.Series):
        return pd.Series(aux_func(X.values, method, param), index=X.index, name=X.name)
    elif isinstance(X, pd.DataFrame):
        return pd.DataFrame(aux_func(X.values, method, param), index=X.index, columns=X.columns)


def _normalize(X, useStd):
    assert isinstance(X, np.ndarray)
    n_axis = len(X.shape)
    n_stocks = X.shape[-1]
    x_mean = np.nanmean(X, axis=n_axis-1, keepdims=True)
    s_std = np.nanstd(X, axis=n_axis-1, keepdims=True)
    if useStd:
        return (X - np.repeat(x_mean, n_stocks, axis=n_axis-1)) / np.repeat(s_std, n_stocks, axis=n_axis-1)
    else:
        return X - np.repeat(x_mean, n_stocks, axis=n_axis-1)


def normalize(X, useStd=True):
    aux_func = _normalize
    if isinstance(X, np.ndarray):
        return aux_func(X, useStd)
    elif isinstance(X, pd.Series):
        return pd.Series(aux_func(X.values, useStd), index=X.index, name=X.name)
    elif isinstance(X, pd.DataFrame):
        return pd.DataFrame(aux_func(X.values, useStd), index=X.index, columns=X.columns)


# # Group Operators
# ---
# Should be optimized
def _group(X, groups, agg_func):
    assert X.shape == groups.shape
    assert len(X.shape) == 2
    result = np.empty_like(X)
    if isinstance(X, pd.DataFrame):
        for i in range(len(X)):
            x, group = X.iloc[i], groups.iloc[i]
            result[i] = (x - x.groupby(group).transform(agg_func))
        return pd.DataFrame(result, index=X.index, columns=X.columns)
    elif isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
        groups = pd.DataFrame(groups)
        for i in range(len(X)):
            x, group = X.iloc[i], groups.iloc[i]
            result[i] = (x - x.groupby(group).transform(agg_func))
        return result


def group_backfill(x, grou, d, std=4):
    pass


def group_count(x, group):
    pass


def group_extra(x, weight, group):
    pass


def group_mean(x, group):
    return _group(x, group, np.mean)


def group_max(x, group):
    return _group(x, group, np.max)


def group_median(x, group):
    return _group(x, group, np.median)


def group_min(x, group):
    return _group(x, group, np.min)


# def _group(x, group, agg_func):
#     if np.sum(np.isnan(x)) + np.sum(np.isnan(group)) > 0:
#         nan_mask = np.isnan(x) | np.isnan(group)
#
#         x_copy = np.where(nan_mask, 0, x)
#         group_copy = np.where(nan_mask, '0', group)
#
#         labels, indices = np.unique(group_copy, return_inverse=True)
#         grouped_val = np.array([agg_func(x_copy[group_copy == label]) for label in labels])
#         rlt = grouped_val[indices]
#         rlt = np.where(nan_mask, np.nan, rlt)
#     else:
#         labels, indices = np.unique(group, return_inverse=True)
#         grouped_val = np.array([agg_func(x[group == label]) for label in labels])
#         rlt = grouped_val[indices]
#     return rlt


def _group_neutralize_single(x, group):
    # Ensure x and group have same length
    assert len(x) == len(group), "Series x and group must have same length."

    # Calculate the mean of each group
    group_means = x.groupby(group).transform('mean')
    # Subtract the mean of each group from the corresponding elements in x
    neutralized_values = x - group_means

    return neutralized_values


def group_neutralize(x, group):
    return sub(x, group_mean(x, group))


def group_rank(x, group):
    pass


def group_sum(x, group):
    return _group(x, group, np.sum)


def group_zscore(x, group):
    pass


# Transformational Operators

def arc_cos(x):
    return np.arccos(x)


def arc_sin(x):
    return np.arc_sin(x)


def arc_tan(x):
    return np.arctan(x)


# def bucket(rank(x), range="0, 1, 0.1" or buckets = "2,5,6,7,10")

# def clamp(x, lower = 0, upper = 0, inverse = False, mask = ")

def filter(x, h = "1, 2, 3, 4", t="0.5"):
    pass


def keep(x, f, period=5):
    pass


def left_tail(x, maximum=0.02):
    """
    left_tail(rank(close), maximum = 0.02)
    """
    return if_else(more(x, maximum), np.nan, x)


def right_tail(x, minimum=0.5):
    """
    right_tail(rank(volume), minimum = 0.5)
    """
    return if_else(less(x, minimum), np.nan, x)


def sigmoid(x):
    return divide(1, add(1, np.exp(-x)))


def tanh(x):
    return np.tanh(x)


def tail(x, lower=0, upper=0.5, newval=np.nan):
    cond = logical_or(less(x, lower), more(x, upper))
    return if_else(cond, x, np.nan)


# TODO: 保持变量类型封闭
def trade_when(trigger: np.ndarray[bool],
               alpha: np.ndarray[np.float64],
               exit_cond: np.ndarray[bool]):
    rlt = if_else(exit_cond, np.nan, alpha)
    rlt = if_else(trigger, rlt, ts_delay(rlt, 1))
    if isinstance(alpha, np.ndarray):
        return rlt
    elif isinstance(alpha, pd.Series):
        return pd.Series(rlt, index=alpha.index, name=alpha.name)
    elif isinstance(alpha, pd.DataFrame):
        return pd.DataFrame(rlt, index=alpha.index, columns=alpha.columns)
