import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..config import __OP_MODE__


# CrossSectionalOperators
def cs_corr(X, Y, method="pearson"):
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.DataFrame)
    assert X.shape == Y.shape
    return X.corrwith(Y, axis=1, method=method)


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


def normalize(X, useStd=True):
    assert isinstance(X, pd.DataFrame)
    if useStd:
        return X.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    else:
        return X.apply(lambda x: (x - x.mean()), axis=1)


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


def _group_neutralize_df(X, Groups):
    # Flatten the dataframes and create a new dataframe with three columns: 'values', 'groups', 'original_index'
    df = pd.DataFrame({
        'values': X.values.flatten(),
        'groups': Groups.values.flatten(),
        'original_index': np.arange(X.size)  # Keep track of the original index
    })

    # Calculate group means
    group_means = df.groupby('groups')['values'].transform('mean')
    # Subtract the group mean from the values
    df['values'] = df['values'] - group_means

    # Reshape the neutralized values to the original shape and convert back to a dataframe
    neutralized_X = pd.DataFrame(df['values'].values.reshape(X.shape), index=X.index, columns=X.columns)

    return neutralized_X


def _group_neutralize_single(x, group):
    # Ensure x and group have same length
    assert len(x) == len(group), "Series x and group must have same length."

    # Calculate the mean of each group
    group_means = x.groupby(group).transform('mean')
    # Subtract the mean of each group from the corresponding elements in x
    neutralized_values = x - group.map(group_means)

    return neutralized_values


def group_neutralize(X, groups):
    assert type(X) == type(groups)
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
