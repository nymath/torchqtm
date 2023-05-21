# Out model is consistent with the official model.
# In this example, we check the two dimentional symbolic regression
from quant.autoalpha.gplearn import fitness
from quant.autoalpha.gplearn.genetic import SymbolicRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
import graphviz


rng = check_random_state(0)

# Training samples
x0 = rng.normal(0, 1, (252, 50))
x1 = rng.normal(2, 0.5, (252, 50))
X = np.concatenate([x0.reshape(252, 50, 1), x1.reshape(252, 50, 1)], axis=2)
y = x0**2 - x1**2 + x1 - 1


def score_func_basic(y, y_pred, sample_weight):
    return np.sqrt(np.sum((y-y_pred)**2))


m = fitness.make_fitness(function=score_func_basic,
                         greater_is_better=False,
                         wrap=False)


est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           metric=m,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1, n_jobs=4,
                           parsimony_coefficient=0.01, random_state=0)

est_gp.fit(X, y)
print(est_gp._program)
dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph