from abc import ABCMeta, abstractmethod

from torchqtm.assets import Equity, Future

from torchqtm.constants import (
    DEFAULT_PER_SHARE_COST,
    DEFAULT_PER_DOLLAR_COST,
    DEFAULT_MINIMUM_COST_PER_EQUITY_TRADE,
    DEFAULT_MINIMUM_COST_PER_FUTURE_TRADE,
)
from torchqtm.finance.orders.order import Order
from torchqtm.finance.transaction import Transaction


class CommissionModel(object, metaclass=ABCMeta):
    """ABC for commission models

    Commission models are responsible for accepting order/transaction pairs and
    calculating how much commission should be charged to an algorithm's account
    on each transaction.
    """
    allowed_asset_types = (Equity, Future)

    @abstractmethod
    def calculate(self, order: Order, txn: Transaction) -> float:
        """
        Returns
        -------
        amount_charged: float
            The additional commission, in yuans, that we should attribute to this order.
        """
        raise NotImplementedError(f"{self.__name__}.calculate")


class NoCommission(CommissionModel):
    """Model that commissions are free.

    """
    def calculate(self, order: Order, txn: Transaction):
        return 0.0


class EquityCommissionModel(CommissionModel, metaclass=ABCMeta):
    """
    Base class for commission models which only support equities.
    """
    allowed_asset_types = (Equity, )


class FutureCommissionModel(CommissionModel, metaclass=ABCMeta):
    allowed_asset_types = (Future, )


def calculate_per_unit_commission(
        order: Order,
        txn: Transaction,
        cost_per_unit: float,
        initial_commission: float,
        min_trade_cost: float,
):
    """
    If there is a minimum commission:
        If the order hasn't had a commission paid yet, pay the minimum
        commission.

        If the order has paid a commission, start paying additional
        commission once the minimum commission has been reached.

    If there is no minimum commission:
        Pay commission based on number of units in the transaction.
    """
    additional_commission = abs(txn.amount * cost_per_unit)

    if order.commission == 0:
        # no commission paid yet, pay at least the minimum plus a one-time
        # exchange fee.
        return max(min_trade_cost, additional_commission + initial_commission)
    else:
        # we've already paid some commission, so figure out how much we
        # would be paying if we only counted per unit.
        per_unit_total = (
            abs(order.filled * cost_per_unit)
            + additional_commission
            + initial_commission
        )

        if per_unit_total < min_trade_cost:
            # if we haven't hit the minimum threshold yet, don't pay
            # additional commission
            return 0
        else:
            # we've exceeded the threshold, so pay more commission.
            return per_unit_total - order.commission


# TODO: Zipline implementation
class PerShare(EquityCommissionModel):
    """
    Calculates a commission for a transaction based on a per share cost with
    an optional minimum cost per trade.

    Parameters
    ----------
    cost : float, optional
        The amount of commissions paid per share traded. Default is one tenth
        of a cent per share.
    min_trade_cost : float, optional
        The minimum amount of commissions paid per trade. Default is no
        minimum.

    Notes
    -----
    This is zipline's default commission model for equities.
    """

    def __init__(
        self,
        cost: float = DEFAULT_PER_SHARE_COST,
        min_trade_cost: float = DEFAULT_MINIMUM_COST_PER_EQUITY_TRADE,
    ):
        self.cost_per_share = float(cost)
        self.min_trade_cost = min_trade_cost or 0

    def __repr__(self):
        return (
            "{class_name}(cost_per_share={cost_per_share}, "
            "min_trade_cost={min_trade_cost})".format(
                class_name=self.__class__.__name__,
                cost_per_share=self.cost_per_share,
                min_trade_cost=self.min_trade_cost,
            )
        )

    def calculate(self, order, txn):
        return calculate_per_unit_commission(
            order=order,
            txn=txn,
            cost_per_unit=self.cost_per_share,
            initial_commission=0,
            min_trade_cost=self.min_trade_cost,
        )


class PerContract(FutureCommissionModel):
    """
    Calculates a commission for a transaction based on a per contract cost with
    an optional minimum cost per trade.

    Parameters
    ----------
    cost : float or dict
        The amount of commissions paid per contract traded. If given a float,
        the commission for all futures contracts is the same. If given a
        dictionary, it must map root symbols to the commission cost for
        contracts of that symbol.
    exchange_fee : float or dict
        A flat-rate fee charged by the exchange per trade. This value is a
        constant, one-time charge no matter how many contracts are being
        traded. If given a float, the fee for all contracts is the same. If
        given a dictionary, it must map root symbols to the fee for contracts
        of that symbol.
    min_trade_cost : float, optional
        The minimum amount of commissions paid per trade.
    """

    def __init__(
        self,
        cost,
        exchange_fee,
        min_trade_cost=DEFAULT_MINIMUM_COST_PER_FUTURE_TRADE,
    ):
        # # If 'cost' or 'exchange fee' are constants, use a dummy mapping to
        # # treat them as a dictionary that always returns the same value.
        # # NOTE: These dictionary does not handle unknown root symbols, so it
        # # may be worth revisiting this behavior.
        # if isinstance(cost, (int, float)):
        #     self._cost_per_contract = DummyMapping(float(cost))
        # else:
        #     # Cost per contract is a dictionary. If the user's dictionary does
        #     # not provide a commission cost for a certain contract, fall back
        #     # on the pre-defined cost values per root symbol.
        #     self._cost_per_contract = defaultdict(
        #         lambda: DEFAULT_PER_CONTRACT_COST, **cost
        #     )
        #
        # if isinstance(exchange_fee, (int, float)):
        #     self._exchange_fee = DummyMapping(float(exchange_fee))
        # else:
        #     # Exchange fee is a dictionary. If the user's dictionary does not
        #     # provide an exchange fee for a certain contract, fall back on the
        #     # pre-defined exchange fees per root symbol.
        #     self._exchange_fee = merge(
        #         FUTURE_EXCHANGE_FEES_BY_SYMBOL,
        #         exchange_fee,
        #     )
        #
        # self.min_trade_cost = min_trade_cost or 0
        pass

    # def __repr__(self):
    #     if isinstance(self._cost_per_contract, DummyMapping):
    #         # Cost per contract is a constant, so extract it.
    #         cost_per_contract = self._cost_per_contract["dummy key"]
    #     else:
    #         cost_per_contract = "<varies>"
    #
    #     if isinstance(self._exchange_fee, DummyMapping):
    #         # Exchange fee is a constant, so extract it.
    #         exchange_fee = self._exchange_fee["dummy key"]
    #     else:
    #         exchange_fee = "<varies>"
    #
    #     return (
    #         "{class_name}(cost_per_contract={cost_per_contract}, "
    #         "exchange_fee={exchange_fee}, min_trade_cost={min_trade_cost})".format(
    #             class_name=self.__class__.__name__,
    #             cost_per_contract=cost_per_contract,
    #             exchange_fee=exchange_fee,
    #             min_trade_cost=self.min_trade_cost,
    #         )
    #     )

    def calculate(self, order, transaction):
        # root_symbol = order.asset.root_symbol
        # cost_per_contract = self._cost_per_contract[root_symbol]
        # exchange_fee = self._exchange_fee[root_symbol]
        #
        # return calculate_per_unit_commission(
        #     order=order,
        #     transaction=transaction,
        #     cost_per_unit=cost_per_contract,
        #     initial_commission=exchange_fee,
        #     min_trade_cost=self.min_trade_cost,
        # )
        pass


class PerTrade(CommissionModel):
    """
    Calculates a commission for a transaction based on a per trade cost.

    For get_orders that require multiple fills, the full commission is charged to
    the first fill.

    Parameters
    ----------
    cost : float, optional
        The flat amount of commissions paid per equity trade.
    """

    def __init__(self, cost=DEFAULT_MINIMUM_COST_PER_EQUITY_TRADE):
        """
        Cost parameter is the cost of a trade, regardless of share count.
        $5.00 per trade is fairly typical of discount brokers.
        """
        # Cost needs to be floating point so that calculation using division
        # logic does not floor to an integer.
        self.cost = float(cost)

    def __repr__(self):
        return "{class_name}(cost_per_trade={cost})".format(
            class_name=self.__class__.__name__,
            cost=self.cost,
        )

    def calculate(self, order, transaction):
        """
        If the order hasn't had a commission paid yet, pay the fixed
        commission.
        """
        if order.commission == 0:
            # if the order hasn't had a commission attributed to it yet,
            # that's what we need to pay.
            return self.cost
        else:
            # order has already had commission attributed, so no more
            # commission.
            return 0.0


class PerFutureTrade(PerContract):
    """
    Calculates a commission for a transaction based on a per trade cost.

    Parameters
    ----------
    cost : float or dict
        The flat amount of commissions paid per trade, regardless of the number
        of contracts being traded. If given a float, the commission for all
        futures contracts is the same. If given a dictionary, it must map root
        symbols to the commission cost for trading contracts of that symbol.
    """

    def __init__(self, cost=DEFAULT_MINIMUM_COST_PER_FUTURE_TRADE):
        # The per-trade cost can be represented as the exchange fee in a
        # per-contract model because the exchange fee is just a one time cost
        # incurred on the first fill.
        super(PerFutureTrade, self).__init__(
            cost=0,
            exchange_fee=cost,
            min_trade_cost=0,
        )
        self._cost_per_trade = self._exchange_fee

    def __repr__(self):
        if isinstance(self._cost_per_trade, DummyMapping):
            # Cost per trade is a constant, so extract it.
            cost_per_trade = self._cost_per_trade["dummy key"]
        else:
            cost_per_trade = "<varies>"
        return "{class_name}(cost_per_trade={cost_per_trade})".format(
            class_name=self.__class__.__name__,
            cost_per_trade=cost_per_trade,
        )


class PerDollar(EquityCommissionModel):
    """
    Model commissions by applying a fixed cost per dollar transacted.

    Parameters
    ----------
    cost : float, optional
        The flat amount of commissions paid per dollar of equities
        traded. Default is a commission of $0.0015 per dollar transacted.
    """

    def __init__(self, cost=DEFAULT_PER_DOLLAR_COST):
        """
        Cost parameter is the cost of a trade per-dollar. 0.0015
        on $1 million means $1,500 commission (=1M * 0.0015)
        """
        self.cost_per_dollar = float(cost)

    def __repr__(self):
        return "{class_name}(cost_per_dollar={cost})".format(
            class_name=self.__class__.__name__, cost=self.cost_per_dollar
        )

    def calculate(self, order, transaction):
        """
        Pay commission based on dollar value of shares.
        """
        cost_per_share = transaction.price * self.cost_per_dollar
        return abs(transaction.amount) * cost_per_share






