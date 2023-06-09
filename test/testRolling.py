import pandas as pd

import torchqtm.core.window.rolling as F
from torchqtm.utils import Timer
import numpy as np
a = np.random.normal(0, 1, (252 * 10, 5000, 2))
a[..., 1] = -a[..., 0]

def ff(x):
    x1 = x[..., 0]
    x2 = x[..., 1]
    rlt = np.sum(x1, axis=0) + np.sum(x2, axis=0)
    return rlt


ff(a).shape
F.roll_apply(a, 5, ff)


m = pd.DataFrame([1,2,3,np.nan,5,6])
b = [0,0,0, 1,1, np.nan]
m.groupby(b).apply(lambda x: np.mean(x))
from scipy.stats import norm

with Timer():
    F.roll_apply_rank(a, 100, mode="roll_single")

with Timer():
    pd.DataFrame(a).rolling(5).rank()

F.roll_apply_rank(a, 10, mode="roll_single")

with Timer():
    F.roll_apply_rank(a, 20, mode="auto")


with Timer("rolling_apply"):
    print(F.roll_apply(a, 500, lambda x: np.max(x, axis=0))[-3:, :5])

with Timer("rolling_apply"):
    print(F.roll_apply_max(a, 5))


with Timer("rolling_apply"):
    print(pd.DataFrame(a).rolling(500).mean().iloc[-3:, :5])

with Timer("rolling_apply"):
    print(pd.DataFrame(a).rolling(5).max().iloc[-3:, :5])