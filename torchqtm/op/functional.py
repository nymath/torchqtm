import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..config import __OP_MODE__
from typing import overload
from .algos import rank_1d, rank_2d
from scipy.stats import norm
import talib


# Arithmetic Operators


@overload
def abs(x: pd.DataFrame) -> pd.DataFrame: ...


@overload
def abs(x: np.ndarray) -> np.ndarray: ...


def abs(x):
    """absolute value of x"""
    if isinstance(x, pd.DataFrame):
        return x.abs(x)
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
    return 1 / X


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
def if_else(X, input2, input3):
    rlt = np.where(X, input2, input3)
    if isinstance(rlt, np.ndarray):
        return rlt
    elif isinstance(rlt, pd.DataFrame):
        return pd.DataFrame(rlt, index=X.index, columns=X.columns)


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
    def aux_fun(array, window_size):
        return roll_apply_min(array, window_size)

    if isinstance(x, np.ndarray):
        return aux_fun(x, d)
    elif isinstance(x, pd.Series):
        return pd.Series(aux_fun(x.values, d), index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(aux_fun(x.values, d), index=x.index, columns=x.columns)


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
    pass


def hump(x, hump=0.01):
    pass


def hump_decay(x, p=0):
    pass


def inst_tvr(x, d):
    pass


def jump_decay(x, d, sensitivity=0.5, force=0.1):
    pass


def kth_element(x, d, k):
    pass


def last_diff_value(x, d):
    pass


def ts_arg_min(x, d):
    def aux_ts_arg_min(t):
        return np.nanargmin(t[::-1], axis=0)

    return ts_apply(x, d, aux_ts_arg_min)


def ts_arg_max(x, d):
    def aux_ts_arg_max(t):
        return np.nanargmax(t[::-1], axis=0)

    return ts_apply(x, d, aux_ts_arg_max)


def ts_av_diff(x, d):
    return sub(x, ts_mean(x, d))


def ts_backfill(x, d, k=1, ignore="NAN"):
    pass


def ts_co_kurtosis(y, x, d):
    pass


def ts_corr(x, y, d):
    pass


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
    pass


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
    pass


def ts_min_diff(x, d):
    return x - ts_min(x, d)


def ts_min_max_cps(x, d, f=2):
    pass


def ts_min_max_diff(x, d, f=0.5):
    pass


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


@overload
def cs_corr(X: pd.DataFrame, Y: pd.DataFrame, method: str = "pearson") -> pd.Series: ...


@overload
def cs_corr(X: np.ndarray, Y: np.ndarray, method: str = "pearson") -> np.ndarray: ...


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
    result = []
    for i in range(len(Y)):
        y = Y.values[i]
        y_demean = y - np.nanmean(y)
        x = X.values[i]
        x_demean = x - np.nanmean(x)
        residuals = y_demean - (np.nanmean(y_demean * x_demean) / np.nanvar(x_demean)) * x_demean
        result.append(residuals)
    return pd.DataFrame(np.array(result), index=Y.index, columns=Y.columns)


def _regression_neuts(Y, others):
    result = []
    for i in range(len(Y)):
        y = Y.values[i]
        X = np.concatenate([x.values[i].reshape(-1, 1) for x in others], axis=1)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        result.append(residuals)
    return pd.DataFrame(np.array(result), index=Y.index, columns=Y.columns)


def regression_neut(Y, others):
    assert isinstance(Y, pd.DataFrame)
    if not isinstance(others, list):
        return _regression_neut(Y, others)
    else:
        if __OP_MODE__ == "STABLE":
            return _regression_neuts(Y, others)
        else:
            pass


def regression_proj(Y, others):
    return Y - regression_neut(Y, others)


def winsorize(X, method='quantile', param=0.05):
    assert isinstance(X, pd.DataFrame)

    def winsorize_series(series):
        if method == 'quantile':
            lower_bound = series.quantile(param)
            upper_bound = series.quantile(1 - param)
        elif method == 'std':
            mean = np.nanmean(series)
            std = np.nanstd(series)
            lower_bound = mean - std * param
            upper_bound = mean + std * param
        else:
            raise ValueError('method should be either "quantile" or "std"')
        series[:] = np.where(series < lower_bound, lower_bound,
                             np.where(series > upper_bound, upper_bound, series))
        return series

    return X.apply(winsorize_series, axis=1)


@overload
def normalize(X: pd.DataFrame, useStd: bool = True) -> pd.DataFrame: ...


@overload
def normalize(X: np.ndarray, useStd: bool = True) -> np.ndarray: ...


def normalize(X, useStd=True):
    if isinstance(X, pd.DataFrame):
        if useStd:
            return X.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        else:
            return X.apply(lambda x: (x - x.mean()), axis=1)
    elif isinstance(X, np.ndarray):
        x_mean = np.nanmean(X, axis=1, keepdims=True)
        s_std = np.nanstd(X, axis=1, keepdims=True)
        if useStd:
            return (X - x_mean) / s_std
        else:
            return X - x_mean


# # Group Operators

# ---

def group_backfill(x, grou, d, std=4):
    pass


def group_count(x, group):
    pass


def group_extra(x, weight, group):
    pass


def group_max(x, group):
    pass


def group_median(x, group):
    pass


def group_min(x, group):
    pass


def _group(x, group, agg_func):
    """
    agg_func should be robust against nan.
    import numpy as np
    import pandas as pd

    def _group(x, group, agg_func):
        # Masking nan values
        nan_mask_x = np.isnan(x)
        nan_mask_group = np.isnan(group)

        # Replacing nan values
        x_copy = np.where(nan_mask_x, 0, x)
        group_copy = np.where(nan_mask_group, '0', group)

        # Using pandas for efficient grouping and aggregation
        df = pd.DataFrame({'x': x_copy, 'group': group_copy})
        grouped_df = df.groupby('group').agg(agg_func)

        # Mapping aggregated values back to original rawdata
        df['rlt'] = df['group'].map(grouped_df['x'])

        # Restoring nan values
        rlt = np.where(nan_mask_x | nan_mask_group, np.nan, df['rlt'].values)

        return rlt
    """
    if np.sum(np.isnan(x)) + np.sum(np.isnan(group)):
        nan_mask = np.isnan(x) | np.isnan(group)

        x_copy = np.where(nan_mask, 0, x)
        group_copy = np.where(nan_mask, '0', group)

        labels, indices = np.unique(group_copy, return_inverse=True)
        grouped_val = np.array([agg_func(x_copy[group_copy == label]) for label in labels])
        rlt = grouped_val[indices]
        rlt = np.where(nan_mask, np.nan, rlt)
    else:
        labels, indices = np.unique(group, return_inverse=True)
        grouped_val = np.array([agg_func(x[group == label]) for label in labels])
        rlt = grouped_val[indices]
    return rlt


def _group_neutralize_single(x, group):
    # Ensure x and group have same length
    assert len(x) == len(group), "Series x and group must have same length."

    # Calculate the mean of each group
    group_means = x.groupby(group).transform('mean')
    # Subtract the mean of each group from the corresponding elements in x
    neutralized_values = x - group_means

    return neutralized_values


def group_neutralize(X, groups):
    assert X.shape == groups.shape
    result = []
    if isinstance(X, pd.DataFrame):
        for i in range(len(X)):
            x, group = X.iloc[i], groups.iloc[i]
            result.append(x - x.groupby(group).transform('mean'))
        return pd.DataFrame(np.array(result), index=X.index, columns=X.columns)
    elif isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
        groups = pd.DataFrame(groups)
        for i in range(len(X)):
            # x, group = X[i], groups[i]
            # result.append(x - _group(x, group, np.nanmean))
            x, group = X.iloc[i], groups.iloc[i]
            result.append(x - x.groupby(group).transform('mean'))
        return np.array(result)


def group_rank(x, group):
    pass


def group_sum(x, group):
    pass


def group_zscore(x, group):
    pass


# Transformational Operators


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


