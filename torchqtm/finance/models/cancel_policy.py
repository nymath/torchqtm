from abc import ABCMeta, abstractmethod

from torchqtm.edbt.gens.sim_engine import Event


class CancelPolicy(object, metaclass=ABCMeta):

    @abstractmethod
    def should_cancel(self, event: Event):
        raise NotImplementedError


class EODCancel(CancelPolicy):
    """
    This policy cancels open get_orders at the end of the day.
    It is highly recommended that only apply this policy to minutely simulations.
    """
    def __init__(self, warn_on_cancel: bool = True):
        self.warn_on_cancel = warn_on_cancel

    def should_cancel(self, event: Event):
        return event == Event.SESSION_END


class NeverCancel(CancelPolicy):
    """
    Order are never automatically canceled.
    """
    def __init__(self):
        self.warn_on_cancel = False

    def should_cancel(self, event: Event):
        return False

