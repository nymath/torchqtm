import pandas as pd
import numpy as np
from torchqtm.op.functional import *
import time
X_n = np.random.normal(0, 1, (252, 50))
X_n[0, [1, 3, 5]] = np.nan
X_n[2, [1, 3, 5]] = np.nan
X_n[4, [1, 3, 5]] = np.nan
idx = pd.date_range('20100101', periods=252)
symbols = [str(i) for i in range(1, 51)]
X_d = pd.DataFrame(X_n, index=idx, columns=symbols)


class Timer(object):
    def __init__(self, project_name=None):
        self.project_name = project_name
        self.start = time.time()

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("{}, Time elapsed: {:.4f}ms".format(self.project_name, 1000*(time.time() - self.start)))


class Tester(object):
    @staticmethod
    def t_ceiling():
        print(ceiling(X_d))

    @staticmethod
    def t_exp():
        print(np.exp(X_d))

    @staticmethod
    def t_nan_out():
        print(nan_out(X_d))


class CSTester(object):
    @ staticmethod
    def t_cs_corr_0():
        with Timer():
            print(cs_corr(X_d, X_d**2, method='spearman'))

    @ staticmethod
    def t_cs_corr_1():
        def corr(X, Y):
            mean_x = np.nanmean(X, axis=1, keepdims=True)
            std_x = np.nanstd(X, axis=1, keepdims=True)
            mean_y = np.nanmean(Y, axis=1, keepdims=True)
            std_y = np.nanstd(Y, axis=1, keepdims=True)
            return np.nanmean((X-mean_x)/std_x * (Y-mean_y)/std_y, axis=1)
        with Timer():
            print(corr(X_d, X_d**2))

        with Timer():
            print(cs_corr(X_n, X_n**3, method='spearman'))


import numpy as np
import pandas as pd
import time




