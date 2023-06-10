import pandas as pd
import torch
import torchqtm.op.functional as F
getattr(F, "ts_mean")

F['ts_mean']
from torchqtm.utils import Timer
import numpy as np
import pandas as pd
a = np.random.normal(0, 1, (252 * 10, 50000))
b = np.random.normal(0, 1, (252 * 100, 500))
c = 2*b + 1


with Timer():
    a = a.astype(np.float64)

am = pd.DataFrame(a)

with Timer():
    am.rolling(20).corr(am)

with Timer():
    F.ts_corr(a, a, 20)


F.ts_corr(a, a, 20)

am.rolling(20).mean().tail(5)

F.ts_mean(a, 20)[-5:]

