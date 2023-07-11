from torchqtm.types import ASSET_TYPE


class Asset:
    type: ASSET_TYPE = ...

    def __init__(self,
                 symbol: str,  # For example "000001.XSHE"
                 asset_name: str = None,
                 start_date: int = None,
                 first_traded: int = None,
                 tick_size: float = 0.01):

        self.symbol: str = symbol
        self.asset_name: str = asset_name
        self.start_date: int = start_date  # Numerical-based
        self.first_traded: int = first_traded  # Numerical-based
        self.tick_size: float = tick_size
        self.sid: int = self.__int__()
        self.price_multiplier: float = 1.

    def __int__(self):
        number_string = self.symbol.split(".")[0]
        # convert the string to an integer
        number = int(number_string)
        return number

    def __repr__(self):
        if self.symbol:
            return f"<{type(self).__name__}({self.symbol})>"

    def __eq__(self, other):
        return self.symbol == other.symbol

    # TODO: implement it
    def is_exchange_open(self):
        pass

    def __hash__(self):
        return hash(self.symbol)


class Equity(Asset):
    type = ASSET_TYPE.Equity

    @classmethod
    def from_int(cls, int_id):
        raise NotImplementedError


class Future(Asset):
    type = ASSET_TYPE.Future


class Index(Asset):
    type = ASSET_TYPE.Index


class Fund(Asset):
    type = ASSET_TYPE.Fund

