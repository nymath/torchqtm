import pandas as pd


class QuantDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return QuantDataFrame

    # Override existing methods or add new methods here
    def annualized_return(self):
        """
        Calculates the annualized return of the time series.
        Assumes the values represent daily asset prices.
        """
        total_return = self.iloc[-1] / self.iloc[0] - 1
        num_years = len(self) / 252  # Approximate number of business days in a year
        return (1 + total_return) ** (1 / num_years) - 1

    def annualized_volatility(self):
        """
        Calculates the annualized volatility (standard deviation of returns) of the time series.
        Assumes the values represent daily asset prices.
        """
        returns = self.pct_change().dropna()
        return returns.std() * (252 ** 0.5)  # Annualize daily volatility

    # You can add more methods here...


class QuantTable(object):
    pass

