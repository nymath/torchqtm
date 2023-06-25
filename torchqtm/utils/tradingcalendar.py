import pandas as pd

start = pd.Timestamp('1990-01-01')
end_base = pd.Timestamp('today')
# Give an aggressive buffer for logic that needs to use the next trading
# day or minute.
end = end_base + pd.Timedelta(days=365)


def canonicalize_datetime(dt):
    # Strip out any HHMMSS or timezone info in the user's datetime, so that
    # all the datetimes we return will be 00:00:00 UTC.
    return datetime(dt.year, dt.month, dt.day, tzinfo=None)


