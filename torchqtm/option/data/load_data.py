import tushare as ts
import pandas as pd
from torchqtm.config import __TS_API__
import mibian
pro = ts.pro_api(__TS_API__)
exchanges = ['SHFE', 'SH']
df = pro.opt_basic(exchange='CFFEX', fields='ts_code,name,exercise_type,list_date,delist_date')

ts.pro_bar(ts_code='000001.SZ', adj='qfq', start_date='20180101', end_date='20181011')