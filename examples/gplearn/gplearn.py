import os
import numpy as np
from quant.autoalpha.gplearn import fitness
from quant.vbt.rebalance import Calendar, Weekly
from quant.autoalpha.gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from quant.vbt.rebalance import Calendar, Weekly
from quant.visualization.visualization import ColorGenerator
from quant.vbt.universe import StaticUniverse, IndexComponents
from quant.vbt.benchmark import BenchMark
from quant.vbt.backtest import BackTestEnv, GPTestingIC
import quant.op as op
import quant.op.functional as F
import matplotlib.pyplot as plt
import time
import graphviz
import pickle


def score_func_basic(y, y_pred, sample_weight, strategy: GPTestingIC):# 适应度函数：策略评价指标
    rlt = strategy.run_backtest(y_pred)
    if np.isnan(rlt):
        return -99
    else:
        return rlt


if __name__ == '__main__':
    start = '20170101'
    end = '20230101'
    calendar = Calendar(start, end)
    universe = StaticUniverse(IndexComponents('000905.SH', start).data)

    with open("../largedata/Stocks.pkl", "rb") as f:
        dfs = pickle.load(f)
    # Create the backtest environment
    BtEnv = BackTestEnv(dfs=dfs,
                        dates=calendar.trade_dates,
                        symbols=universe.symbols)

    bt = GPTestingIC(env=BtEnv,
                     universe=universe)

    m = fitness.make_fitness(function=score_func_basic,
                             greater_is_better=True,
                             wrap=False)

    features = ['Open', 'High', 'Low', 'Close']

    symbolic_model = SymbolicRegressor(population_size=40,
                                       generations=6,
                                       tournament_size=20,
                                       metric=m,
                                       function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                                     'abs', 'neg', 'inv', 'sin', 'cos', 'tan',
                                                     'max', 'min'),
                                       feature_names=features,
                                       const_range=None,
                                       parsimony_coefficient='auto',
                                       stopping_criteria=100,
                                       init_depth=(2, 6),
                                       init_method='half and half',
                                       p_crossover=0.4,
                                       p_subtree_mutation=0.01,
                                       p_hoist_mutation=0.0,
                                       p_point_mutation=0.01,
                                       p_point_replace=0.4,
                                       max_samples=1,
                                       n_jobs=4,
                                       verbose=1,
                                       warm_start=False,
                                       low_memory=False,
                                       random_state=0,
                                       strategy=bt)

    fixed_shape = bt.env.Open.values.shape
    x1 = bt.env.Open.values.reshape(*fixed_shape, 1)
    x2 = bt.env.High.values.reshape(*fixed_shape, 1)
    x3 = bt.env.Low.values.reshape(*fixed_shape, 1)
    x4 = bt.env.Close.values.reshape(*fixed_shape, 1)
    x5 = bt.env.Volume.values.reshape(*fixed_shape, 1)

    X = np.concatenate((x1, x2, x3, x4), axis=2)
    y = x1  # Have no role in the fitness function
    symbolic_model.fit(X, y)
    print(symbolic_model._program)
    idx = symbolic_model._program.parents['donor_idx']
    fade_nodes = symbolic_model._program.parents['donor_nodes']
    dot_data = symbolic_model._programs[-2][idx].export_graphviz(fade_nodes=fade_nodes)
    graph = graphviz.Source(dot_data)