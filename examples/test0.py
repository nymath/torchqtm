import pandas as pd

import torchqtm._libs.window.aggregations as F
from torchqtm.utils import Timer

import numpy as np

a = np.random.normal(0, 1, 1000000)
end = np.arange(1, len(a), 1, dtype=np.int64)
start = end - 5
end = np.clip(end, 0, len(a))
start = np.clip(start, 0, len(a))
F.roll_sum(a, start, end, 5)
# F.roll_sum(a, )
with Timer("apply"):
    F.roll_apply(a, start, end, 5, lambda x: np.sum(x), True, (), {})

with Timer("apply"):
    pd.DataFrame(a).rolling(5).apply(lambda x: np.sum(x))

with Timer("sum"):
    F.roll_sum(a, start, end, 5)



