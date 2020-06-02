import numpy as np

from enum import Enum
from typing import *

flat_arr_type = Union[list, np.ndarray]
raw_data_type = Union[List[List[Union[str, float]]], None]
data_type = Union[raw_data_type, np.ndarray]


class DataTuple:
    def __init__(self,
                 x: data_type,
                 y: data_type):
        self.x, self.y = x, y

    def __eq__(self, other: "DataTuple"):
        if not np.allclose(self.x, other.x, equal_nan=True):
            return False
        if self.y is None and other.y is not None:
            return False
        if self.y is not None and other.y is None:
            return False
        if self.y is None and other.y is None:
            return True
        return np.allclose(self.y, other.y, equal_nan=True)

    def __str__(self):
        if isinstance(self.x, np.ndarray):
            with np.printoptions(precision=3, suppress=True):
                return str(np.hstack([self.x, self.y]))
        return str([feature + label for feature, label in zip(self.x, self.y)])

    __repr__ = __str__

    @property
    def xT(self):
        if getattr(self, "_xt", None) is None:
            if isinstance(self.x, list):
                self._xt = list(map(list, zip(*self.x)))
            else:
                self._xt = self.x.T
        return self._xt


class ColumnTypes(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    STRING = "string"


class TaskTypes(Enum):
    REGRESSION = "reg"
    CLASSIFICATION = "clf"

    @classmethod
    def from_column_type(cls,
                         column_type: ColumnTypes) -> "TaskTypes":
        if column_type is ColumnTypes.NUMERICAL:
            return cls.REGRESSION
        return cls.CLASSIFICATION


class FeatureInfo(NamedTuple):
    contains_nan: bool
    flat_arr: Union[flat_arr_type, None]
    is_valid: bool = True
    nan_mask: np.ndarray = None
    need_transform: bool = None
    column_type: ColumnTypes = ColumnTypes.NUMERICAL
    unique_values_sorted_by_counts: np.ndarray = None
    msg: str = None

    @property
    def is_string(self) -> bool:
        return self.column_type is ColumnTypes.STRING

    @property
    def is_categorical(self) -> bool:
        return self.column_type is ColumnTypes.CATEGORICAL

    @property
    def is_numerical(self) -> bool:
        return self.column_type is ColumnTypes.NUMERICAL


__all__ = [
    "flat_arr_type", "raw_data_type", "data_type",
    "DataTuple", "ColumnTypes", "TaskTypes", "FeatureInfo"
]
