import pandas as pd
from typing import Optional, Union, Iterable, Dict, Hashable


class TqFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return TqFrame

    def get_rows(self, row_index: Union[int, Iterable[int]]):
        return self.iloc[row_index]


