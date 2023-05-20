import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats.mstats import winsorize as Winsorize


# CrossSectionalOperators
def cs_corr(X, Y, method='pearson'):
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
        return _regression_neuts(Y, others)


def regression_proj(Y, others):
    return Y - regression_neut(Y, others)


def winsorize(X, alpha=0.05):
    assert isinstance(X, pd.DataFrame)
    return X.apply(lambda x: Winsorize(x, limits=[alpha, alpha]), axis=0)


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
    assert isinstance(X, pd.DataFrame)
    assert X.shape == groups.shape
    result = []
    for i in range(len(X)):
        x, group = X.iloc[i], groups.iloc[i]
        result.append(x - x.groupby(group).transform('mean'))
    return pd.DataFrame(np.array(result), index=X.index, columns=X.columns)


def group_rank(x, group):
    pass


def group_sum(x, group):
    pass


def group_zscore(x, group):
    pass


