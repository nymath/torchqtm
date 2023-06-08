import pandas as pd

import torchqtm.core.window.rolling as F
from torchqtm.utils import Timer
import numpy as np
a = np.random.normal(0, 1, (252 * 10, 5000))

with Timer("rolling_apply"):
    print(F.roll_apply(a, 5, lambda x: np.max(x, axis=0))[-3:, :5])

with Timer("rolling_apply"):
    print(F.roll_apply_max(a, 5))


with Timer("rolling_apply"):
    print(pd.DataFrame(a).rolling(5).max().iloc[-3:, :5])
