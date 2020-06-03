import numpy as np

from enum import Enum
from typing import *

flat_arr_type = Union[list, np.ndarray]
raw_data_type = Union[List[List[Union[str, float]]], None]
data_type = Union[raw_data_type, np.ndarray]


class DataTuple(NamedTuple):
    x: data_type
    y: data_type
    xT: data_type = None

    def __eq__(self, other: "DataTuple"):
        self_x_is_list = isinstance(self.x, list)
        other_x_is_list = isinstance(other.x, list)
        if self_x_is_list and not other_x_is_list:
            return False
        if not self_x_is_list and other_x_is_list:
            return False
        if self_x_is_list:
            x_allclose = self.x == other.x
        else:
            if isinstance(self.x[0][0], np.str_):
                x_allclose = self.x.tolist() == other.x.tolist()
            else:
                x_allclose = np.allclose(self.x, other.x, equal_nan=True)
        if not x_allclose:
            return False
        if self.y is None and other.y is not None:
            return False
        if self.y is not None and other.y is None:
            return False
        if self.y is None and other.y is None:
            return True
        self_y_is_list = isinstance(self.y, list)
        other_y_is_list = isinstance(other.y, list)
        if self_y_is_list and not other_y_is_list:
            return False
        if not self_y_is_list and other_y_is_list:
            return False
        if self_y_is_list:
            return self.y == other.y
        if isinstance(self.y[0][0], np.str_):
            return self.y.tolist() == other.y.tolist()
        return np.allclose(self.y, other.y, equal_nan=True)

    def __ne__(self, other: "DataTuple"):
        return not self == other

    @property
    def xy(self) -> Tuple[data_type, data_type]:
        return self.x, self.y

    @classmethod
    def with_transpose(cls,
                       x: data_type,
                       y: data_type):
        xt = x.T if not isinstance(x, list) else list(map(list, zip(*x)))
        return DataTuple(x, y, xt)


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
