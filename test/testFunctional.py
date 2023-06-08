import pandas as pd

import torchqtm.op.functional as F
from torchqtm.utils import Timer
import numpy as np
import pandas as pd
a = np.random.normal(0, 1, (252 * 10, 5000))
b = np.random.normal(0, 1, (252 * 10, 50))

F.ts_rank(b, 5)


with Timer():
    F.ts_arg_min(a, 5)

b = np.array([[3,np.nan], [2, 5], [1, 2]])
b




with Timer():
    F.ts_rank(a, 20, mode='overall')
with Timer():
    F.ts_rank(a, 50, mode='single')



with Timer():
    F.ts_apply(a, 5, lambda x: np.max(x, axis=0))


with Timer():
    pd.DataFrame(a).rolling(5).apply(lambda x: np.max(x, axis=0))

