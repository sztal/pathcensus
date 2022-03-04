"""Custom type defintions."""
# pylint: disable=unused-import
from typing import Union
import numpy as np
import pandas as pd
from .graph import GraphABC    # type: ignore

Data = Union[pd.Series, pd.DataFrame]

UInt  = np.uint
Float = np.dtype(float).type
