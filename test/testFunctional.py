import torchqtm.op.functional as F
from torchqtm.utils import Timer
import numpy as np
a = np.random.normal(0, 1, (252 * 10, 5000))

with Timer():
    F.ts_max(a, 5)

with Timer():
    F.ts_apply(a, 5, lambda x: np.max(x, axis=0))
