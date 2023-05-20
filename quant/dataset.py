# 设计一个自己的数据集, 他能够维持




# import pandas as pd
# import numpy as np
# from ogalpha.torchquantum.utils import Calendar
# from ogalpha.torchquantum.op.functional import *
# from sklearn.linear_model import LinearRegression
# from collections import Iterable
# import copy
# import time
#
# testdata = {}
# calendar = Calendar('20100101', '20150101')
# trade_dates = calendar.trade_dates
# _ = calendar.create_weekly_groups()
# rebalance_dates = [x[-1] for x in _.values()]
#
# dfs = {}
# columns = pd.RangeIndex(range(1, 5001), dtype=int)
#
# dfs['Close'] = pd.DataFrame(np.random.normal(5000, 1000, (len(trade_dates), len(columns))), index=trade_dates,
#                             columns=columns)
# dfs['mv'] = pd.DataFrame(np.random.normal(1000, 100, (len(trade_dates), len(columns))), index=trade_dates,
#                          columns=columns)
# dfs['pb'] = pd.DataFrame(np.random.normal(1, 0.1, (len(trade_dates), len(columns))), index=trade_dates, columns=columns)
#
# temp = pd.DataFrame(np.random.randint(1, 31, (1, len(columns))), columns=columns)
# dfs['sector'] = pd.concat([temp for i in range(len(trade_dates))], axis=0)
# dfs['sector'].index = trade_dates
#
# used_datas = {key: x.loc[rebalance_dates] for key, x in dfs.items()}
# used_datas['future_return'] = used_datas['Close'].pct_change(-1)


# winsorize

# winsorize(used_datas['pb'], 0.05)

# standardize
# normalize(used_datas['pb']) # 目前效率很低

# neutralize

# import statsmodels.api as sm
# group_neutralize(used_datas['pb'], used_datas['sector'])
# regression_neut(used_datas['pb'], used_datas['sector'])
#
# print("Hello world!")


# # class BackTestDataSet(object):
# #     def __init__(self, data):
# #         assert isinstance(data, np.ndarray)
# #         self.data = data
# #
# #     def __len__(self):
# #         return len(self.data)
# #
# #     @property
# #     def shape(self):
# #         return self.data.shape
# #
# #     def __getitem__(self, *args, **kwargs):
# #         """改进为二重索引"""
# #         return self.data.__getitem__(*args, **kwargs)
# #
# #     def __repr__(self):
# #         return self.data.__repr__()
# #
# #     def _cs_project(self, others):
# #         """
# #         others 可以是一个BackTestDataSet的迭代器
# #         """
# #         assert isinstance(others, BackTestDataSet)
# #         if not isinstance(others, list):
# #             others = [others]
# #         assert all([len(x) == len(self) for x in others])
# #         result = []
# #         for i in range(len(self)):
# #             y = self.data.__getitem__(i)
# #             X = np.concatenate([x.data.__getitem__(i).reshape(-1, 1) for x in others], axis=1)
# #             model = LinearRegression()
# #             model.fit(X, y)
# #             y_pred = model.predict(X)
# #             residuals = y - y_pred
# #             result.append(residuals)
# #         return BackTestDataSet(np.array(result))
# #
# #     def standardize(self):
# #         pass
# #
# #     def winsorize(self):
# #         pass
# #
# #     def neutralize(self):
# #         pass
# #
# # a1 = BackTestDataSet(used_datas['pb'].values)
# # a2 = BackTestDataSet(used_datas['mv'].values)
# # a1._cs_project(a2).shape
#
#
# def regression_neut(Y, others):
#     if not isinstance(others, list):
#         others = [others]
#     assert all([len(Y) == len(x) for x in others])
#     assert isinstance(Y, pd.DataFrame)
#     result = []
#     for i in range(len(Y)):
#         y = Y.values[i]
#         X = np.concatenate([x.values[i].reshape(-1, 1) for x in others], axis=1)
#         model = LinearRegression()
#         model.fit(X, y)
#         y_pred = model.predict(X)
#         residuals = y - y_pred
#         result.append(residuals)
#     return pd.DataFrame(np.array(result), index=Y.index, columns=Y.columns)
#
#
# def regression_proj(Y, others):
#     return Y - regression_neut(Y, others)
#
#
# used_datas['final_pb'] = regression_neut(used_datas['pb'], used_datas['mv'])
#
# # %time _ = used_datas['pb'].iloc[0].groupby(used_datas['sector'].iloc[0]).mean()
#
# def group_neutralize(x, group):
#     # Ensure x and group have same length
#     assert len(x) == len(group), "Lists x and group must have same length."
#
#     # Identify unique groups
#     unique_groups = set(group)
#
#     # Initialize a list to hold the neutralized values
#     neutralized_values = []
#
#     # For each unique group...
#     for g in unique_groups:
#         # ...identify the indices of elements in this group...
#         indices = [i for i, x in enumerate(group) if x == g]
#
#         # ...calculate the mean of the values in this group...
#         mean = sum(x[i] for i in indices) / len(indices)
#
#         # ...and subtract this mean from each value in the group, appending the result to neutralized_values.
#         neutralized_values.extend(x[i] - mean for i in indices)
#
#     return neutralized_values
#
#
# # %time _ = group_neutralize(used_datas['pb'].iloc[0], used_datas['sector'].iloc[0])
#
#
# def group_neutralize_row(row):
#     x = row[:len(row)//2]
#     group = row[len(row)//2:]
#     group_means = x.groupby(group).mean()
#     neutralized_values = x - group.map(group_means)
#     return neutralized_values
#
# # Concatenate the dataframes along the second axis
# combined_df = pd.concat([used_datas['pb'], used_datas['sector']], axis=1)
#
# # Apply the function row-wise
# # %time neutralized_df = combined_df.apply(group_neutralize_row, axis=1)
#
# def group_neutralize_df(X, Groups):
#     # Flatten the dataframes and create a new dataframe with three columns: 'values', 'groups', 'original_index'
#     df = pd.DataFrame({
#         'values': X.values.flatten(),
#         'groups': Groups.values.flatten(),
#         'original_index': np.arange(X.size)  # Keep track of the original index
#     })
#
#     # Calculate group means
#     group_means = df.groupby('groups')['values'].transform('mean')
#     # Subtract the group mean from the values
#     df['values'] = df['values'] - group_means
#
#     # Reshape the neutralized values to the original shape and convert back to a dataframe
#     neutralized_X = pd.DataFrame(df['values'].values.reshape(X.shape), index=X.index, columns=X.columns)
#
#     return neutralized_X
#
#
#
#
#
#
#
# # 分组, 计算收益
#
# import time
# k = 5
# labels = ["group_" + str(i + 1) for i in range(k)]  # 为每个组创建标签
#
#
# t0 = time.time()
# groups = []
# weights = []
# returns = []
# for i in range(len(used_datas['final_pb'])):
#     group = pd.qcut(used_datas['final_pb'].iloc[i], k, labels=labels[::-1])
#     data = pd.concat([used_datas['mv'].iloc[i], used_datas['future_return'].iloc[i]], axis=1)
#
#     def temp(x):
#         weight = x.iloc[:, 0] / x.iloc[:, 0].sum()
#         weights.append(weight)
#         ret = x.iloc[:, 1]
#         return (weight * ret).sum()
#
#     group_return = data.groupby(group).apply(temp)
#     groups.append(group)
#     returns.append(group_return)
# print(time.time() - t0)
#
#
# def cs_correlation(X, Y, method='pearson'):
#     assert isinstance(X, pd.DataFrame)
#     assert isinstance(Y, pd.DataFrame)
#     assert X.shape == Y.shape
#     return X.corrwith(Y, axis=1, method=method)
#
#
