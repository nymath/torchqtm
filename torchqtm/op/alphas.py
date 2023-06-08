import torchqtm.op as op
import torchqtm.op.functional as F
import pandas as pd
import numpy as np
pd.DataFrame.rolling

a = np.zeros((10, 20), dtype=np.float64)
a = pd.DataFrame(a)


def f(x):
    return np.mean(x)

a.rolling(5).apply(f)


# def algo():
#     lag = 22
#     ldf = {}
#     adjfactor = get_daily_data("AdjFactor",lines=lag)
#     for k in key:
#         ldf[k] = np.log(get_daily_data(k,lines=lag)*adjfactor)
#     ldf['ClosePrice_lag_1'] = np.apply_along_axis(lambda x:shift(x,1,cval=np.nan),0,ldf['ClosePrice'])
#     a = np.sqrt(np.nanmean(0.5*(ldf['HighPrice']-ldf['LowPrice'])**2,axis=0)- \
#         (2*np.log(2)-1)*np.nanmean((ldf['ClosePrice']-ldf['OpenPrice'])**2,axis=0)+ \
#         np.nanmean((ldf['OpenPrice']-ldf['ClosePrice_lag_1'])**2,axis=0))
#
#     return a


class Ross(op.Volatility):
    def __init__(self):
        super().__init__()

    def operate(self):
        Open = F.log(self.env.Open)
        High = F.log(self.env.High)
        Low = F.log(self.env.Low)
        Close = F.log(self.env.Close)
        Closel1 = F.ts_delay(self.env.Close)
