import pandas as pd

start = pd.Timestamp('1990-01-01')
end_base = pd.Timestamp('today')
# Give an aggressive buffer for logic that needs to use the next trading
# day or minute.
end = end_base + pd.Timedelta(days=365)


