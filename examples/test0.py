import pandas as pd

import torchqtm._libs.window.aggregations as F
# import pyximport
# import numpy
# pyximport.install(setup_args={'include_dirs': numpy.get_include()})

import torchqtm._C._rolling as RF

from torchqtm.utils import Timer
import numpy as np
from torchqtm.core.window.rolling import roll_apply

a = np.random.normal(0, 1, (252 * 10, 5000))
end = np.arange(1, len(a)+1, 1, dtype=np.int64)
start = end - 5
end = np.clip(end, 0, len(a))
start = np.clip(start, 0, len(a))
# F.roll_sum(a, start, end, 5)
# F.roll_sum(a, )
with Timer("apply"):
    for i in range(a.shape[1]):
        # F.roll_apply(a[:, i], start, end, 5, lambda x: np.sum(x), True, (), {})
        RF.roll_max(a[:, i], start, end, 5)

#
# with Timer("apply"):
#     for i in range(a.shape[1]):
#         RF.roll_apply(a[:, i], start, end, 5, lambda x: np.sum(x), True, (), {})
#
# with Timer("apply"):
#     for i in range(a.shape[1]):
#         F.roll_apply(a[:, i], start, end, 5, lambda x: np.sum(x), True, (), {})

with Timer('roll_apply_2D'):
    roll_apply(a, 5, lambda x: np.sum(x, axis=0))


with Timer('pd'):
    pd.DataFrame(a).rolling(5).apply(lambda x: np.sum(x))


with Timer("apply"):
    pd.DataFrame(a).rolling(5).sum()


with Timer("sum"):
    F.roll_sum(a, start, end, 5)



