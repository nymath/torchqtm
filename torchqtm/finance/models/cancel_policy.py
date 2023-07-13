from abc import ABCMeta, abstractmethod

from torchqtm.types import EVENT_TYPE


class CancelPolicy(object, metaclass=ABCMeta):

    @abstractmethod
    def should_cancel(self, event: EVENT_TYPE):
        raise NotImplementedError


class EODCancel(CancelPolicy):
    """
    This policy cancels open get_orders at the end of the day.
    It is highly recommended that only apply this policy to minutely simulations.
    """
    def __init__(self, warn_on_cancel: bool = True):
        self.warn_on_cancel = warn_on_cancel

    def should_cancel(self, event: EVENT_TYPE):
        return event == EVENT_TYPE.SESSION_END


class NeverCancel(CancelPolicy):
    """
    Order are never automatically canceled.
    """
    def __init__(self):
        self.warn_on_cancel = False

    def should_cancel(self, event: EVENT_TYPE):
        return False

