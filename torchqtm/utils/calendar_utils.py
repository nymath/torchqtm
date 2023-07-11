from .exchange_calendar import XSHGExchangeCalendar


def get_calendar(*args, **kwargs):
    if args[0] == "XSHG":
        return XSHGExchangeCalendar()
    else:
        return ValueError("Invalid calendar argument")

