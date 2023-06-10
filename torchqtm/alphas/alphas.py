import torchqtm.op as op
import torchqtm.op.functional as F
import pandas as pd
import numpy as np

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

    def forward(self):
        raw_shape = self.env.Open.shape
        Open = np.array(F.log(self.env.Open), dtype=np.float64).reshape(*raw_shape, 1)
        High = np.array(F.log(self.env.High), dtype=np.float64).reshape(*raw_shape, 1)
        Low = np.array(F.log(self.env.Low), dtype=np.float64).reshape(*raw_shape, 1)
        Close = np.array(F.log(self.env.Close), dtype=np.float64).reshape(*raw_shape, 1)
        Closel1 = np.array(F.ts_delay(self.env.Close), dtype=np.float64).reshape(*raw_shape, 1)
        data = np.concatenate([Open, High, Low, Close, Closel1], axis=2)

        def aux_func(data_slice):
            cl = {
                'Open': 0,
                'High': 1,
                'Low': 2,
                'Close': 3,
                'Closel1': 4
            }
            return np.sqrt(np.nanmean(0.5 * (data_slice[..., cl['High']] - data_slice[..., cl['Low']]) ** 2, axis=0) -
                           (2 * np.log(2) - 1) * np.nanmean(
                (data_slice[..., cl['Close']] - data_slice[..., cl['Open']]) ** 2, axis=0) +
                           np.nanmean((data_slice[..., cl['Open']] - data_slice[..., cl['Closel1']]) ** 2, axis=0))
        rlt = F.ts_apply(data, 10, aux_func)
        return rlt
