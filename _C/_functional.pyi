from typing import overload, List

import pandas as pd
import numpy as np


@overload
def regression_neut(Y: pd.DataFrame, others: pd.DataFrame) -> pd.DataFrame: ...
@overload
def regression_neut(Y: np.ndarray, others: np.ndarray) -> np.ndarray: ...
@overload
def regression_neut(Y: pd.DataFrame, others: List[pd.DataFrame, ...]) -> pd.DataFrame: ...
@overload
def regression_neut(Y: np.ndarray, others: List[np.ndarray, ...]) -> np.ndarray: ...

