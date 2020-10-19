import os
import dill
import math
import random

import numpy as np

from typing import *
from cftool.misc import *
from sklearn.datasets import *
from enum import Enum
from abc import abstractmethod, ABCMeta
from functools import partial
from sklearn.utils import Bunch

from ..types import *


# types

flat_arr_type = Union[list, np.ndarray]
raw_data_type = Optional[List[List[Union[str, float]]]]
data_type = Union[raw_data_type, np.ndarray]


def transpose(x: data_type):
    return x.T if isinstance(x, np.ndarray) else list(map(list, zip(*x)))


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

    def split_with(self,
                   indices: Union[np.ndarray, List[int]]) -> "DataTuple":

        def _fetch(arr):
            if arr is None:
                return
            if isinstance(arr, np.ndarray):
                return arr[indices]
            assert isinstance(arr, list)
            return [arr[i] for i in indices]

        x, y = map(_fetch, [self.x, self.y])

        xt = None
        if self.xT is not None:
            if isinstance(self.xT, np.ndarray):
                xt = self.xT[..., indices]
            else:
                xt = [[line[i] for i in indices] for line in self.xT]
        return DataTuple(x, y, xt)

    @classmethod
    def with_transpose(cls,
                       x: data_type,
                       y: data_type):
        return DataTuple(x, y, transpose(x))


class ColumnTypes(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    STRING = "string"


class TaskTypes(Enum):
    NONE = ""
    REGRESSION = "reg"
    CLASSIFICATION = "clf"
    TIME_SERIES_CLF = "ts_clf"
    TIME_SERIES_REG = "ts_reg"

    @property
    def is_none(self) -> bool:
        return self is TaskTypes.NONE

    @property
    def is_clf(self) -> bool:
        return self is TaskTypes.CLASSIFICATION or self is TaskTypes.TIME_SERIES_CLF

    @property
    def is_reg(self) -> bool:
        return self is TaskTypes.REGRESSION or self is TaskTypes.TIME_SERIES_REG

    @property
    def is_ts(self) -> bool:
        return self is TaskTypes.TIME_SERIES_CLF or self is TaskTypes.TIME_SERIES_REG

    @classmethod
    def from_str(cls,
                 task_type: str) -> "TaskTypes":
        if task_type == "":
            return cls.NONE
        if task_type == "reg":
            return cls.REGRESSION
        if task_type == "clf":
            return cls.CLASSIFICATION
        if task_type == "ts_clf":
            return cls.TIME_SERIES_CLF
        if task_type == "ts_reg":
            return cls.TIME_SERIES_REG
        raise ValueError(f"task_type '{task_type}' is not recognized")

    @classmethod
    def from_column_type(cls,
                         column_type: ColumnTypes,
                         *,
                         is_time_series: bool) -> "TaskTypes":
        if column_type is ColumnTypes.NUMERICAL:
            return cls.TIME_SERIES_REG if is_time_series else cls.REGRESSION
        return cls.TIME_SERIES_CLF if is_time_series else cls.CLASSIFICATION


class FeatureInfo(NamedTuple):
    contains_nan: Optional[bool]
    flat_arr: Optional[flat_arr_type]
    is_valid: bool = True
    nan_mask: Optional[np.ndarray] = None
    need_transform: Optional[bool] = None
    column_type: ColumnTypes = ColumnTypes.NUMERICAL
    unique_values_sorted_by_counts: Optional[np.ndarray] = None
    msg: Optional[str] = None

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
    task_type: TaskTypes = TaskTypes.NONE
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
        return self.task_type.is_clf

    @property
    def is_reg(self) -> bool:
        return self.task_type.is_reg

    @property
    def is_ts(self) -> bool:
        return self.task_type.is_ts

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
        y = y.reshape([-1, 1]).astype(np_int_type if task_type.is_clf else np_float_type)
        return x, y

    def split_with(self,
                   indices: np.ndarray) -> "TabularDataset":
        return TabularDataset(self.x[indices], self.y[indices], *self[2:])

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
        if task_type.is_reg:
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


# utils

class DataStructure(LoggingMixin, metaclass=ABCMeta):
    core_file = "core.pkl"

    @abstractmethod
    def dumps(self) -> bytes:
        pass

    @abstractmethod
    def loads(self, instance_dict: Dict[str, Any], **kwargs: Any) -> "SavingMixin":
        pass

    def dump(self,
             folder: str,
             *,
             compress: bool = True,
             remove_original: bool = True) -> None:
        abs_folder = os.path.abspath(folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [folder]):
            Saving.prepare_folder(self, abs_folder)
            with open(os.path.join(abs_folder, self.core_file), "wb") as f:
                f.write(self.dumps())
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)

    @classmethod
    def load(cls,
             *,
             data: Optional[bytes] = None,
             folder: Optional[str] = None,
             compress: bool = True,
             **kwargs: Any) -> "DataStructure":
        if data is not None:
            instance_dict = dill.loads(data)
            return cls.loads(instance_dict, **kwargs)
        if folder is None:
            raise ValueError("either `folder` or `data` should be provided")
        base_folder = os.path.dirname(os.path.abspath(folder))
        with lock_manager(base_folder, [folder]):
            with Saving.compress_loader(
                folder,
                compress,
                remove_extracted=True,
            ):
                with open(os.path.join(folder, cls.core_file), "rb") as f:
                    return cls.loads(dill.load(f), **kwargs)


def split_file(file: str,
               export_folder: str,
               *,
               has_header: bool = None,
               indices_pair: Tuple[Iterable[int], Iterable[int]] = None,
               split: Union[int, float] = 0.1) -> Tuple[str, str]:
    os.makedirs(export_folder, exist_ok=True)
    with open(file, "r") as f:
        data = f.readlines()
    ext = os.path.splitext(file)[1]
    if has_header is None:
        has_header = ext == ".csv"
    header = None
    if has_header:
        header, data = data[0], data[1:]
    split1 = os.path.join(export_folder, f"split1{ext}")
    split2 = os.path.join(export_folder, f"split2{ext}")

    if indices_pair is None:
        num_data = len(data)
        indices = list(range(num_data))
        if split < 1. or split == 1. and isinstance(split, float):
            split = int(num_data * split)
        random.shuffle(indices)
        indices_pair = (indices[:split], indices[split:])

    def _split(file_, indices_):
        with open(file_, "w") as f:
            if header is not None:
                f.write(header)
            for idx in indices_:
                f.write(data[idx])

    _split(split1, indices_pair[0])
    _split(split2, indices_pair[1])

    return split1, split2


class SplitResult(NamedTuple):
    dataset: TabularDataset
    corresponding_indices: np.ndarray
    remaining_indices: Optional[np.ndarray]

    @classmethod
    def concat(cls,
               results: List["SplitResult"]) -> "SplitResult":
        datasets = [result.dataset for result in results]
        basic_info = datasets[0][2:]
        x_list, y_list = zip(*[dataset.xy for dataset in datasets])
        x_concat, y_concat = map(np.vstack, [x_list, y_list])
        return SplitResult(
            TabularDataset(x_concat, y_concat, *basic_info),
            np.hstack([result.corresponding_indices for result in results]),
            np.hstack([result.remaining_indices for result in results])
        )


class TimeSeriesConfig(NamedTuple):
    id_column_name: str = None
    time_column_name: str = None
    id_column_idx: int = None
    time_column_idx: int = None
    id_column: np.ndarray = None
    time_column: np.ndarray = None


class DataSplitter(SavingMixin):
    """
    Util class for dividing dataset based on task type
    * If it's regression task, it's simple to split data
    * If it's classification task, we need to split data based on labels, because we need
    to ensure the divided data contain all labels available

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cfdata.types import np_int_type
    >>> from cfdata.tabular.misc import TaskTypes, DataSplitter
    >>> from cfdata.tabular.wrapper import TabularDataset
    >>>
    >>> x = np.arange(12).reshape([6, 2])
    >>> # create an imbalance dataset
    >>> y = np.zeros(6, np_int_type)
    >>> y[[-1, -2]] = 1
    >>> dataset = TabularDataset.from_xy(x, y, TaskTypes.CLASSIFICATION)
    >>> data_splitter = DataSplitter().fit(dataset)
    >>> # labels in result will keep its ratio
    >>> result = data_splitter.split(3)
    >>> # [0 0 1]
    >>> print(result.dataset.y.ravel())
    >>> data_splitter.reset()
    >>> result = data_splitter.split(0.5)
    >>> # [0 0 1]
    >>> print(result.dataset.y.ravel())
    >>> # at least one sample of each class will be kept
    >>> y[-2] = 0
    >>> dataset = TabularDataset.from_xy(x, y, TaskTypes.CLASSIFICATION)
    >>> data_splitter = DataSplitter().fit(dataset)
    >>> result = data_splitter.split(2)
    >>> # [0 0 0 0 0 1] [0 1]
    >>> print(y, result.dataset.y.ravel())

    """

    @property
    def data_tuple_base(self) -> Optional[Type[NamedTuple]]:
        return

    @property
    def data_tuple_attributes(self) -> Optional[List[str]]:
        return

    def __init__(self,
                 *,
                 time_series_config: Optional[TimeSeriesConfig] = None,
                 shuffle: bool = True,
                 replace: bool = False,
                 verbose_level: int = 2):
        self._remained_indices = None
        self._time_indices_list = self._time_indices_list_in_use = None
        self._label_indices_list = self._label_indices_list_in_use = None
        self._time_series_config, self._time_series_sorting_indices = time_series_config, None
        self._shuffle, self._replace = shuffle, replace
        self._verbose_level = verbose_level
        if time_series_config is not None:
            if replace:
                raise ValueError("`replace` cannot be True when splitting time series dataset")
            self._id_column = time_series_config.id_column
            self._time_column = time_series_config.time_column
            self._id_column_idx = time_series_config.id_column_idx
            self._time_column_idx = time_series_config.time_column_idx
            if self._id_column is None and self._id_column_idx is None:
                raise ValueError("either `id_column` or `id_column_idx` should be provided")
            if self._time_column is None and self._time_column_idx is None:
                raise ValueError("either `time_column` or `time_column_idx` should be provided")

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def id_column(self):
        return self._id_column

    @property
    def time_column(self):
        return self._time_column

    @property
    def sorting_indices(self):
        if not self._dataset.is_ts:
            raise ValueError("sorting_indices should not be called when it is not time series condition")
        return self._time_series_sorting_indices

    @property
    def remained_indices(self):
        return self._remained_indices[::-1].copy()

    @property
    def remained_xy(self):
        indices = self.remained_indices
        return self._x[indices], self._y[indices]

    # reset methods

    def _reset_reg(self):
        num_data = len(self._x)
        if not self._shuffle:
            self._remained_indices = np.arange(num_data)
        else:
            self._remained_indices = np.random.permutation(num_data)
        self._remained_indices = self._remained_indices.astype(np_int_type)

    def _reset_clf(self):
        if self._label_indices_list is None:
            flattened_y = self._y.ravel()
            unique_indices = get_unique_indices(flattened_y)
            self._unique_labels, counts = unique_indices[:2]
            self._label_indices_list = unique_indices.split_indices
            self._num_samples = len(flattened_y)
            self._label_ratios = counts / self._num_samples
            self._num_unique_labels = len(self._unique_labels)
            if self._num_unique_labels == 1:
                raise ValueError("only 1 unique label is detected, which is invalid in classification task")
            self._unique_labels = self._unique_labels.astype(np_int_type)
            self._label_indices_list = list(map(partial(np.asarray, dtype=np_int_type), self._label_indices_list))
        self._reset_indices_list("label_indices_list")

    def _reset_time_series(self):
        if self._time_indices_list is None:
            self.log_msg(f"gathering time -> indices mapping", self.info_prefix, verbose_level=5)
            unique_indices = get_unique_indices(self._time_column)
            self._unique_times = unique_indices.unique[::-1]
            times_counts = unique_indices.unique_cnt[::-1]
            self._time_indices_list = unique_indices.split_indices[::-1]
            self._times_counts_cumsum = np.cumsum(times_counts).astype(np_int_type)
            assert self._times_counts_cumsum[-1] == len(self._time_column)
            self._time_series_sorting_indices = np.hstack(self._time_indices_list[::-1]).astype(np_int_type)
            self._time_indices_list = list(map(partial(np.asarray, dtype=np_int_type), self._time_indices_list))
        self._reset_indices_list("time_indices_list")
        self._times_counts_cumsum_in_use = self._times_counts_cumsum.copy()

    def _reset_indices_list(self, attr):
        self_attr = getattr(self, f"_{attr}")
        if self._shuffle:
            tuple(map(np.random.shuffle, self_attr))
        attr_in_use = f"_{attr}_in_use"
        setattr(self, attr_in_use, [arr.copy() for arr in self_attr])
        self._remained_indices = np.hstack(getattr(self, attr_in_use)).astype(np_int_type)

    # split methods

    def _split_reg(self, n: int):
        tgt_indices = self._remained_indices[-n:]
        n = min(n, len(self._remained_indices) - 1)
        if self._replace:
            np.random.shuffle(self._remained_indices)
        elif n > 0:
            self._remained_indices = self._remained_indices[:-n]
        return tgt_indices

    def _split_clf(self, n: int):
        if n < self._num_unique_labels:
            raise ValueError(
                f"at least {self._num_unique_labels} samples are required because "
                f"we have {self._num_unique_labels} unique labels"
            )
        pop_indices_list, tgt_indices_list = [], []
        num_samples_per_label = np.maximum(1, np.round(n * self._label_ratios).astype(np_int_type))
        # -num_unique_labels <= num_samples_exceeded <= num_unique_labels
        num_samples_exceeded = num_samples_per_label.sum() - n
        # adjust n_samples_per_label to make sure `n` samples are split out
        if num_samples_exceeded != 0:
            sign, num_samples_exceeded = np.sign(num_samples_exceeded), abs(num_samples_exceeded)
            chosen_indices = np.arange(self._num_unique_labels)[num_samples_per_label != 1]
            np.random.shuffle(chosen_indices)
            num_chosen_indices = len(chosen_indices)
            num_tile = int(np.ceil(num_samples_exceeded / num_chosen_indices))
            num_proceeded = 0
            for _ in range(num_tile - 1):
                num_samples_per_label[chosen_indices] -= sign
                num_proceeded += num_chosen_indices
            for idx in chosen_indices[:num_samples_exceeded - num_proceeded]:
                num_samples_per_label[idx] -= sign
        assert num_samples_per_label.sum() == n
        num_overlap = 0
        for indices, num_sample_per_label in zip(self._label_indices_list_in_use, num_samples_per_label):
            num_samples_in_use = len(indices)
            tgt_indices_list.append(indices[-num_sample_per_label:])
            if num_sample_per_label >= num_samples_in_use:
                pop_indices_list.append([])
                num_overlap += num_sample_per_label
            else:
                pop_indices_list.append(np.arange(num_samples_in_use - num_sample_per_label, num_samples_in_use))
        tgt_indices = np.hstack(tgt_indices_list)
        if self._replace:
            tuple(map(np.random.shuffle, self._label_indices_list_in_use))
            self._remained_indices = np.hstack(self._label_indices_list_in_use)
        else:
            self._label_indices_list_in_use = list(map(
                lambda arr, pop_indices: np.delete(arr, pop_indices),
                self._label_indices_list_in_use, pop_indices_list
            ))
            remain_indices = np.hstack(self._label_indices_list_in_use)
            base = np.zeros(self._num_samples)
            base[tgt_indices] += 1
            base[remain_indices] += 1
            assert np.sum(base >= 2) <= num_overlap
            self._remained_indices = remain_indices
        return tgt_indices

    def _split_time_series(self, n: int):
        split_arg = np.argmax(self._times_counts_cumsum_in_use >= n)
        num_left = self._times_counts_cumsum_in_use[split_arg] - n
        if split_arg == 0:
            num_res, selected_indices = n, []
        else:
            num_res = n - self._times_counts_cumsum_in_use[split_arg - 1]
            selected_indices = self._time_indices_list_in_use[:split_arg]
            self._time_indices_list_in_use = self._time_indices_list_in_use[split_arg:]
            self._times_counts_cumsum_in_use = self._times_counts_cumsum_in_use[split_arg:]
        selected_indices.append(self._time_indices_list_in_use[0][:num_res])
        if num_left > 0:
            self._time_indices_list_in_use[0] = self._time_indices_list_in_use[0][num_res:]
        else:
            self._time_indices_list_in_use = self._time_indices_list_in_use[1:]
            self._times_counts_cumsum_in_use = self._times_counts_cumsum_in_use[1:]
        tgt_indices, remained_indices = map(np.hstack, [selected_indices, self._time_indices_list_in_use])
        self._times_counts_cumsum_in_use -= n
        self._remained_indices = remained_indices[::-1].copy()
        return tgt_indices[::-1].copy()

    def fit(self,
            dataset: TabularDataset) -> "DataSplitter":
        self._dataset = dataset
        self._x = dataset.x
        self._y = dataset.y
        if not self._dataset.is_ts:
            self._time_column = None
        else:
            if self._id_column is None or self._time_column is None:
                id_idx, time_idx = self._id_column_idx, self._time_column_idx
                if self._id_column_idx < self._time_column_idx:
                    id_first = True
                    split_list = [id_idx, id_idx + 1, time_idx, time_idx + 1]
                else:
                    id_first = False
                    split_list = [time_idx, time_idx + 1, id_idx, id_idx + 1]
                columns = np.split(self._x, split_list, axis=1)
                if id_first:
                    self._id_column, self._time_column = columns[1], columns[3]
                else:
                    self._id_column, self._time_column = columns[3], columns[1]
                self._x = np.hstack([columns[0], columns[2], columns[4]])
            self._id_column, self._time_column = map(np.ravel, [self._id_column, self._time_column])
        return self.reset()

    def reset(self) -> "DataSplitter":
        if self._dataset.is_ts:
            self._reset_time_series()
        elif self._dataset.is_reg:
            self._reset_reg()
        else:
            self._reset_clf()
        return self

    def split(self,
              n: Union[int, float]) -> SplitResult:
        error_msg = "please call 'reset' method before calling 'split' method"
        if self._dataset.is_ts:
            if self._time_indices_list_in_use is None:
                raise ValueError(error_msg)
        else:
            if self._dataset.is_reg and self._remained_indices is None:
                raise ValueError(error_msg)
            if self._dataset.is_clf and self._label_indices_list_in_use is None:
                raise ValueError(error_msg)
        if n >= len(self._remained_indices):
            remained_x, remained_y = self.remained_xy
            return SplitResult(
                TabularDataset.from_xy(remained_x, remained_y, self._dataset.task_type),
                self._remained_indices, np.array([], np_int_type)
            )
        if n < 1. or (n == 1. and isinstance(n, float)):
            n = int(round(len(self._x) * n))
        if self._dataset.is_ts:
            split_method = self._split_time_series
        else:
            split_method = self._split_reg if self._dataset.is_reg else self._split_clf
        tgt_indices = split_method(n)
        assert len(tgt_indices) == n
        return SplitResult(self._dataset.split_with(tgt_indices), tgt_indices, self._remained_indices)

    def split_multiple(self,
                       n_list: List[Union[int, float]],
                       *,
                       return_remained: bool = False) -> List[SplitResult]:
        n_list = n_list.copy()
        n_total = len(self._x)
        if not all(n_ <= 1. for n_ in n_list):
            if any(n_ < 1. for n_ in n_list):
                raise ValueError("some of the elements in `n_list` (but not all) are less than 1")
            if return_remained:
                n_list.append(n_total - sum(n_list))
        else:
            ratio_sum = sum(n_list)
            if ratio_sum > 1.:
                raise ValueError("sum of `n_list` should not be greater than 1")
            if return_remained and ratio_sum == 1:
                raise ValueError("sum of `n_list` should be less than 1 "
                                 "when `return_remained` is True")
            n_selected = int(round(n_total * ratio_sum))
            n_list[:-1] = [int(round(n_total * ratio)) for ratio in n_list[:-1]]
            n_list[-1] = n_selected - sum(n_list[:-1])
            if ratio_sum < 1.:
                n_list.append(n_total - n_selected)
        return list(map(self.split, n_list))


__all__ = [
    "flat_arr_type", "raw_data_type", "data_type",
    "transpose", "DataTuple", "ColumnTypes", "TaskTypes", "FeatureInfo", "TabularDataset",
    "split_file", "SplitResult", "TimeSeriesConfig", "DataSplitter",
    "DataStructure",
]
