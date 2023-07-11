from torchqtm.alphas.autoalpha.gplearn import fitness
from torchqtm.alphas.autoalpha.gplearn import SymbolicRegressor
import numpy as np


def score_func_basic(y, y_pred, sample_weight):  # 适应度函数：策略评价指标
    rlt = np.sum((y - y_pred) ** 2)
    return rlt


if __name__ == '__main__':
    X = np.random.normal(0, 1, size=(252, 50, 6))
    y_ture = X[:, :, -1]

    m = fitness.make_fitness(function=score_func_basic,
                             greater_is_better=False,
                             wrap=False)
    symbolic_model = SymbolicRegressor(population_size=200,
                                       generations=6,
                                       tournament_size=20,
                                       metric=m,
                                       function_set=('add', 'sub'),
                                       feature_names=None,
                                       const_range=(-5000, 5000),
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
                                       n_jobs=1,
                                       verbose=1,
                                       warm_start=False,
                                       low_memory=False,
                                       random_state=0)

    symbolic_model.fit(X, y_ture)