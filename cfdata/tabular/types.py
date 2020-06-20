import math
import numpy as np

from typing import *
from enum import Enum
from functools import partial
from sklearn.utils import Bunch
from sklearn.datasets import *

from ..types import *

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

    def split_with(self, indices: np.ndarray) -> "DataTuple":
        if self.xT is None:
            x = xT = None
        elif isinstance(self.xT, np.ndarray):
            x = None
            xT = self.xT[..., indices]
        else:
            x = [self.x[i] for i in indices]
            y = [self.y[i] for i in indices]
            xT = [[line[i] for i in indices] for line in self.xT]
        if x is None:
            x, y = self.x[indices], self.y[indices]
        return DataTuple(x, y, xT)

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
    def from_str(cls,
                 task_type: str) -> "TaskTypes":
        if task_type == "reg":
            return cls.REGRESSION
        if task_type == "clf":
            return cls.CLASSIFICATION
        raise ValueError(f"task_type '{task_type}' is not recognized")

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


class TabularDataset(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    task_type: Union[None, TaskTypes] = None
    label_name: Union[None, str] = "label"
    label_names: Union[None, List[str]] = None
    feature_names: Union[None, List[str]] = None

    def __len__(self):
        return self.x.shape[0]

    @property
    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    @property
    def is_clf(self) -> bool:
        return self.task_type is TaskTypes.CLASSIFICATION

    @property
    def is_reg(self) -> bool:
        return self.task_type is TaskTypes.REGRESSION

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def num_classes(self) -> int:
        if self.task_type == "reg":
            return 0
        return self.y.max().item() + 1

    @staticmethod
    def to_task_type(x: np.ndarray,
                     y: np.ndarray,
                     task_type: TaskTypes) -> Tuple[np.ndarray, np.ndarray]:
        x = x.astype(np_float_type)
        is_clf = task_type is TaskTypes.CLASSIFICATION
        y = y.reshape([-1, 1]).astype(np_int_type if is_clf else np_float_type)
        return x, y

    @classmethod
    def from_bunch(cls,
                   bunch: Bunch,
                   task_type: TaskTypes) -> "TabularDataset":
        x, y = TabularDataset.to_task_type(bunch.data, bunch.target, task_type)
        label_names = bunch.get("target_names")
        feature_names = bunch.get("feature_names")
        return TabularDataset(x, y, task_type, label_names=label_names, feature_names=feature_names)

    @classmethod
    def from_xy(cls,
                x: np.ndarray,
                y: np.ndarray,
                task_type: TaskTypes) -> "TabularDataset":
        x, y = cls.to_task_type(x, y, task_type)
        return TabularDataset(x, y, task_type)

    # scikit-learn datasets

    @classmethod
    def iris(cls) -> "TabularDataset":
        return cls.from_bunch(load_iris(), TaskTypes.CLASSIFICATION)

    @classmethod
    def boston(cls) -> "TabularDataset":
        return cls.from_bunch(load_boston(), TaskTypes.REGRESSION)

    @classmethod
    def digits(cls) -> "TabularDataset":
        return cls.from_bunch(load_digits(), TaskTypes.CLASSIFICATION)

    @classmethod
    def breast_cancer(cls) -> "TabularDataset":
        return cls.from_bunch(load_breast_cancer(), TaskTypes.CLASSIFICATION)

    # artificial datasets

    @classmethod
    def xor(cls, *,
            size: int = 100,
            scale: float = 1.) -> "TabularDataset":
        x = np.random.randn(size) * scale
        y = np.random.randn(size) * scale
        z = (x * y >= 0).astype(np_int_type)
        return TabularDataset.from_xy(np.c_[x, y].astype(np_float_type), z, TaskTypes.CLASSIFICATION)

    @classmethod
    def spiral(cls, *,
               size: int = 50,
               scale: float = 4.,
               nun_spirals: int = 7,
               num_classes: int = 7) -> "TabularDataset":
        xs = np.zeros((size * nun_spirals, 2), dtype=np_float_type)
        ys = np.zeros(size * nun_spirals, dtype=np_int_type)
        pi = math.pi
        for i in range(nun_spirals):
            ix = range(size * i, size * (i + 1))
            r = np.linspace(0.0, 1, size + 1)[1:]
            t_start = 2 * i * pi / nun_spirals
            t_end = 2 * (i + scale) * pi / nun_spirals
            t = np.linspace(t_start, t_end, size) + np.random.random(size=size) * 0.1
            xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            ys[ix] = i % num_classes
        return cls.from_xy(xs, ys, TaskTypes.CLASSIFICATION)

    @classmethod
    def two_clusters(cls, *,
                     size: int = 100,
                     scale: float = 1.,
                     center: float = 0.,
                     distance: float = 2.,
                     num_dimensions: int = 2) -> "TabularDataset":
        center1 = (np.random.random(num_dimensions) + center - 0.5) * scale + distance
        center2 = (np.random.random(num_dimensions) + center - 0.5) * scale - distance
        cluster1 = (np.random.randn(size, num_dimensions) + center1) * scale
        cluster2 = (np.random.randn(size, num_dimensions) + center2) * scale
        data = np.vstack((cluster1, cluster2)).astype(np_float_type)
        labels = np.array([1] * size + [0] * size, np_int_type)
        indices = np.random.permutation(size * 2)
        data, labels = data[indices], labels[indices]
        return cls.from_xy(data, labels, TaskTypes.CLASSIFICATION)

    @classmethod
    def simple_non_linear(cls, *,
                          size: int = 120) -> "TabularDataset":
        xs = np.random.randn(size, 2).astype(np_float_type) * 1.5
        ys = np.zeros(size, dtype=np_int_type)
        mask = xs[..., 1] >= xs[..., 0] ** 2
        xs[..., 1][mask] += 2
        ys[mask] = 1
        return cls.from_xy(xs, ys, TaskTypes.CLASSIFICATION)

    @classmethod
    def nine_grid(cls, *,
                  size: int = 120) -> "TabularDataset":
        x, y = np.random.randn(2, size).astype(np_float_type)
        labels = np.zeros(size, np_int_type)
        xl, xr = x <= -1, x >= 1
        yf, yc = y <= -1, y >= 1
        x_mid_mask = ~xl & ~xr
        y_mid_mask = ~yf & ~yc
        mask2 = x_mid_mask & y_mid_mask
        labels[mask2] = 2
        labels[(x_mid_mask | y_mid_mask) & ~mask2] = 1
        xs = np.vstack([x, y]).T
        return cls.from_xy(xs, labels, TaskTypes.CLASSIFICATION)

    @staticmethod
    def _fetch_ys(affine_train, affine_test, task_type):
        if task_type is TaskTypes.REGRESSION:
            y_train, y_test = affine_train, affine_test
        else:
            y_train = (affine_train > 0).astype(np_int_type)
            y_test = (affine_test > 0).astype(np_int_type)
        return y_train, y_test

    @classmethod
    def noisy_linear(cls, *,
                     size: int = 10000,
                     n_dim: int = 100,
                     n_valid: int = 5,
                     noise_scale: float = 0.5,
                     task_type: TaskTypes = TaskTypes.REGRESSION,
                     test_ratio: float = 0.15) -> Tuple["TabularDataset", "TabularDataset"]:
        x_train = np.random.randn(size, n_dim)
        x_train_noise = x_train + np.random.randn(size, n_dim) * noise_scale
        x_test = np.random.randn(int(size * test_ratio), n_dim)
        idx = np.random.permutation(n_dim)[:n_valid]
        w = np.random.randn(n_valid, 1)
        affine_train = x_train[..., idx].dot(w)
        affine_test = x_test[..., idx].dot(w)
        tr_set, te_set = map(
            cls.from_xy,
            [x_train_noise, x_test],
            TabularDataset._fetch_ys(affine_train, affine_test, task_type),
            2 * [task_type]
        )
        return tr_set, te_set

    @classmethod
    def noisy_poly(cls, *,
                   p: int = 3,
                   size: int = 10000,
                   n_dim: int = 100,
                   n_valid: int = 5,
                   noise_scale: float = 0.5,
                   task_type: TaskTypes = TaskTypes.REGRESSION,
                   test_ratio: float = 0.15) -> Tuple["TabularDataset", "TabularDataset"]:
        assert p > 1, "p should be greater than 1"
        x_train = np.random.randn(size, n_dim)
        x_train_list = [x_train] + [x_train ** i for i in range(2, p + 1)]
        x_train_noise = x_train + np.random.randn(size, n_dim) * noise_scale
        x_test = np.random.randn(int(size * test_ratio), n_dim)
        x_test_list = [x_test] + [x_test ** i for i in range(2, p + 1)]
        idx_list = [np.random.permutation(n_dim)[:n_valid] for _ in range(p)]
        w_list = [np.random.randn(n_valid, 1) for _ in range(p)]
        o_train = [x[..., idx].dot(w) for x, idx, w in zip(x_train_list, idx_list, w_list)]
        o_test = [x[..., idx].dot(w) for x, idx, w in zip(x_test_list, idx_list, w_list)]
        affine_train, affine_test = map(partial(np.sum, axis=0), [o_train, o_test])
        tr_set, te_set = map(
            cls.from_xy,
            [x_train_noise, x_test],
            TabularDataset._fetch_ys(affine_train, affine_test, task_type),
            2 * [task_type]
        )
        return tr_set, te_set


__all__ = [
    "flat_arr_type", "raw_data_type", "data_type",
    "DataTuple", "ColumnTypes", "TaskTypes", "FeatureInfo", "TabularDataset"
]
