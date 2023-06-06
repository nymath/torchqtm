import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..config import __OP_MODE__
from typing import overload
from .algos import rank_1d, rank_2d
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
    return X / Y


def exp(X):
    if isinstance(X, np.ndarray):
        return np.exp(X)
    elif isinstance(X, pd.DataFrame):
        return X.apply(np.exp, axis=1)


def inverse(X):
    return 1 / X


def log(X):
    return np.log(X)


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
def if_else(X, input2, input3):
    assert X.shape == input2.shape == input3.shape
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
def ts_mean(x, d):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.rolling(d).mean()
    elif isinstance(x, np.ndarray):
        return pd.DataFrame(x).rolling(d).mean().values
    else:
        raise TypeError("Input should be a pandas DataFrame, Series or a numpy ndarray.")


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
    if np.sum(np.isnan(x))+np.sum(np.isnan(group)):
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
