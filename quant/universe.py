# import tushare as ts
# pro = ts.pro_api("23d1a24b48f32db03ea94cc67985f95232ec0daf7ee7f066fe2d27de")
#
# df = pro.index_weight(index_code='399300.SZ', start_date='20180901', end_date='20180930')


class StaticUniverse(object):
    def __init__(self, symbols):
        self.symbols = symbols

    def get_symbols(self, trade_date=None):
        return self.symbols


class DynamicUniverse(object):
    pass