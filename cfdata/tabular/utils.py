import os
import random

import numpy as np

from typing import *
from cftool.misc import *
from functools import partial
from abc import ABCMeta, abstractmethod

from .types import *
from ..types import *


def split_file(file: str,
               export_folder: str,
               *,
               has_header: bool = None,
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

    num_data = len(data)
    indices = list(range(num_data))
    if split < 1. or split == 1. and isinstance(split, float):
        split = int(num_data * split)
    random.shuffle(indices)

    def _split(file_, indices_):
        with open(file_, "w") as f:
            if header is not None:
                f.write(header)
            for idx in indices_:
                f.write(data[idx])

    _split(split1, indices[:split])
    _split(split2, indices[split:])

    return split1, split2


class SplitResult(NamedTuple):
    dataset: TabularDataset
    corresponding_indices: np.ndarray
    remaining_indices: Union[np.ndarray, None]

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
    >>> from cfdata.tabular.types import TaskTypes
    >>> from cfdata.tabular.wrapper import TabularDataset
    >>> from cfdata.tabular.utils import DataSplitter
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
    def data_tuple_base(self) -> Union[None, Type[NamedTuple]]:
        return

    @property
    def data_tuple_attributes(self) -> Union[None, List[str]]:
        return

    def __init__(self,
                 *,
                 time_series_config: TimeSeriesConfig = None,
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
        tgt_indices, self._remained_indices = map(np.hstack, [selected_indices, self._time_indices_list_in_use])
        self._times_counts_cumsum_in_use -= n
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
                self._remained_indices, np.array([], np.int)
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


class KFold:
    """
    Util class which can perform k-fold data splitting:

    1. X = {x1, x2, ..., xn} -> [X1, X2, ..., Xk]
    2. Xi ∩ Xj = ∅, ∀ i, j = 1,..., K
    3. X1 ∪ X2 ∪ ... ∪ Xk = X

    * Notice that `KFold` does not always hold the principles listed above, because `DataSplitter`
    will ensure that at least one sample of each class will be kept. In this case, when we apply
    `KFold` to an imbalance dataset, `KFold` may slightly violate principle 2. and 3.

    Parameters
    ----------
    k : int, number of folds
    dataset : TabularDataset, dataset which we want to split
    **kwargs : used to initialize `DataSplitter` instance

    Examples
    ----------
    >>> import numpy as np
    >>>
    >>> from cfdata.types import np_int_type
    >>> from cfdata.tabular.types import TaskTypes
    >>> from cfdata.tabular.wrapper import TabularDataset
    >>> from cfdata.tabular.utils import KFold
    >>>
    >>> x = np.arange(12).reshape([6, 2])
    >>> # create an imbalance dataset
    >>> y = np.zeros(6, np_int_type)
    >>> y[[-1, -2]] = 1
    >>> dataset = TabularDataset.from_xy(x, y, TaskTypes.CLASSIFICATION)
    >>> k_fold = KFold(3, dataset)
    >>> for train_fold, test_fold in k_fold:
    >>>     print(np.vstack([train_fold.dataset.x, test_fold.dataset.x]))
    >>>     print(np.vstack([train_fold.dataset.y, test_fold.dataset.y]))

    """

    def __init__(self,
                 k: int,
                 dataset: TabularDataset,
                 **kwargs):
        if k <= 1:
            raise ValueError("k should be larger than 1 in KFold")
        ratio = 1. / k
        self.n_list = (k - 1) * [ratio]
        self.splitter = DataSplitter(**kwargs).fit(dataset)
        self.split_results = self._order = self._cursor = None

    def __iter__(self):
        self.split_results = self.splitter.split_multiple(self.n_list, return_remained=True)
        self._order = np.random.permutation(len(self.split_results)).tolist()
        self._cursor = 0
        return self

    def __next__(self) -> Tuple[SplitResult, SplitResult]:
        if self._cursor >= len(self._order):
            raise StopIteration
        train_results = self.split_results.copy()
        test_result = train_results.pop(self._order[self._cursor])
        train_result = SplitResult.concat(train_results)
        self._cursor += 1
        return train_result, test_result


class KRandom:
    """
    Util class which can perform k-random data splitting:

    1. X = {x1, x2, ..., xn} -> [X1, X2, ..., Xk]
    2. idx{X1} ≠ idx{X2} ≠ ... ≠ idx{Xk}, where idx{X} = {1, 2, ..., n}
    3. X1 = X2 = ... = Xk = X

    Parameters
    ----------
    k : int, number of folds
    num_test : {int, float}
    * if float and  < 1 : ratio of the test dataset
    * if int   and  > 1 : exact number of test samples
    dataset : TabularDataset, dataset which we want to split
    **kwargs : used to initialize `DataSplitter` instance

    Examples
    ----------
    >>> import numpy as np
    >>>
    >>> from cfdata.types import np_int_type
    >>> from cfdata.tabular.types import TaskTypes
    >>> from cfdata.tabular.wrapper import TabularDataset
    >>> from cfdata.tabular.utils import KRandom
    >>>
    >>> x = np.arange(12).reshape([6, 2])
    >>> # create an imbalance dataset
    >>> y = np.zeros(6, np_int_type)
    >>> y[[-1, -2]] = 1
    >>> dataset = TabularDataset.from_xy(x, y, TaskTypes.CLASSIFICATION)
    >>> k_random = KRandom(3, 2, dataset)
    >>> for train_fold, test_fold in k_random:
    >>>     print(np.vstack([train_fold.dataset.x, test_fold.dataset.x]))
    >>>     print(np.vstack([train_fold.dataset.y, test_fold.dataset.y]))

    """

    def __init__(self,
                 k: int,
                 num_test: Union[int, float],
                 dataset: TabularDataset,
                 **kwargs):
        self._cursor = None
        self.k, self.num_test = k, num_test
        self.splitter = DataSplitter(**kwargs).fit(dataset)

    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self) -> Tuple[SplitResult, SplitResult]:
        if self._cursor >= self.k:
            raise StopIteration
        self._cursor += 1
        self.splitter.reset()
        test_result, train_result = self.splitter.split_multiple([self.num_test], return_remained=True)
        return train_result, test_result


class KBootstrap:
    """
    Util class which can perform k-random data splitting with bootstrap:

    1. X = {x1, x2, ..., xn} -> [X1, X2, ..., Xk] (Use bootstrap aggregation to collect datasets)
    3. idx{X1} ≠ idx{X2} ≠ ... ≠ idx{Xk}, where idx{X} = {1, 2, ..., n}
    4. X1 = X2 = ... = Xk = X

    * Notice that only some of the special algorithms (e.g. bagging) need `KBootstrap`.

    Parameters
    ----------
    k : int, number of folds
    num_test : {int, float}
    * if float and  < 1 : ratio of the test dataset
    * if int   and  > 1 : exact number of test samples
    dataset : TabularDataset, dataset which we want to split
    **kwargs : used to initialize `DataSplitter` instance

    Examples
    ----------
    >>> import numpy as np
    >>>
    >>> from cfdata.types import np_int_type
    >>> from cfdata.tabular.types import TaskTypes
    >>> from cfdata.tabular.wrapper import TabularDataset
    >>> from cfdata.tabular.utils import KBootstrap
    >>>
    >>> x = np.arange(12).reshape([6, 2])
    >>> # create an imbalance dataset
    >>> y = np.zeros(6, np_int_type)
    >>> y[[-1, -2]] = 1
    >>> dataset = TabularDataset.from_xy(x, y, TaskTypes.CLASSIFICATION)
    >>> k_bootstrap = KBootstrap(3, 2, dataset)
    >>> for train_fold, test_fold in k_bootstrap:
    >>>     print(np.vstack([train_fold.dataset.x, test_fold.dataset.x]))
    >>>     print(np.vstack([train_fold.dataset.y, test_fold.dataset.y]))

    """

    def __init__(self,
                 k: int,
                 num_test: Union[int, float],
                 dataset: TabularDataset,
                 **kwargs):
        self._cursor = None
        self.dataset = dataset
        self.num_samples = len(dataset)
        if isinstance(num_test, float):
            num_test = int(round(num_test * self.num_samples))
        self.k, self.num_test = k, num_test
        self.splitter = DataSplitter(**kwargs).fit(dataset)

    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self) -> Tuple[SplitResult, SplitResult]:
        if self._cursor >= self.k:
            raise StopIteration
        self._cursor += 1
        self.splitter.reset()
        test_result, train_result = self.splitter.split_multiple([self.num_test], return_remained=True)
        tr_indices = train_result.corresponding_indices
        tr_indices = np.random.choice(tr_indices, len(tr_indices))
        tr_set = self.dataset.split_with(tr_indices)
        tr_split = SplitResult(tr_set, tr_indices, None)
        return tr_split, test_result


aggregation_dict: Dict[str, Type["AggregationBase"]] = {}


class AggregationBase(LoggingMixin, metaclass=ABCMeta):
    def __init__(self,
                 data,
                 config: Dict[str, Any],
                 verbose_level: int):
        if not data.is_ts:
            raise ValueError("time series data is required")
        self.data = data
        self.config, self._verbose_level = config, verbose_level
        self._num_history = config.setdefault("num_history", 1)
        id_column = data.raw.xT[data.ts_config.id_column_idx]
        unique_indices = get_unique_indices(id_column)
        self._unique_id_arr, self._id2indices = unique_indices.unique, unique_indices.split_indices
        self._initialize()

    @property
    @abstractmethod
    def num_aggregation(self):
        pass

    @abstractmethod
    def _aggregate_core(self, indices: np.ndarray) -> np.ndarray:
        """ indices should be a column vector """

    def _initialize(self):
        self._num_samples_per_id = np.array(list(map(len, self._id2indices)), np.int64)
        self.log_msg("generating valid aggregation info", self.info_prefix, 5)
        valid_mask = self._num_samples_per_id >= self._num_history
        if not valid_mask.any():
            raise ValueError(
                "current settings lead to empty valid dataset, increasing raw dataset size or "
                f"decreasing n_history (current: {self._num_history}) might help"
            )
        if not valid_mask.all():
            invalid_mask = ~valid_mask
            n_invalid_id = invalid_mask.sum()
            n_invalid_samples = self._num_samples_per_id[invalid_mask].sum()
            self.log_msg(
                f"{n_invalid_id} id (with {n_invalid_samples} samples) will be dropped "
                f"(n_history={self._num_history})", self.info_prefix, verbose_level=2
            )
            self.log_msg(
                f"dropped id : {', '.join(map(str, self._unique_id_arr[invalid_mask].tolist()))}",
                self.info_prefix, verbose_level=4
            )
        self._num_samples_per_id_cumsum = np.hstack([[0], np.cumsum(self._num_samples_per_id[:-1])])
        # self._id2indices need to contain 'redundant' indices here because
        # aggregation need to aggregate those 'invalid' samples
        self._id2indices_stack = np.hstack(self._id2indices)
        self.log_msg("generating aggregation attributes", self.info_prefix, verbose_level=5)
        self._get_id2valid_indices()
        self._get_valid_samples_info()

    def _get_valid_samples_info(self):
        # 'indices' in self.indices2id here doesn't refer to indices of original dataset
        # (e.g. 'indices' in self._id2indices), but refers to indices generated by sampler,
        # so we should only care 'valid' indices here
        self._num_valid_samples_per_id = [len(indices) for indices in self._id2valid_indices]
        self._num_valid_samples_per_id_cumsum = np.hstack([[0], np.cumsum(self._num_valid_samples_per_id[:-1])])
        self.indices2id = np.repeat(np.arange(len(self._unique_id_arr)), self._num_valid_samples_per_id)
        self._id2valid_indices_stack = np.hstack(self._id2valid_indices)

    def _get_id2valid_indices(self):
        # TODO : support nan_fill here
        nan_fill, nan_ratio = map(self.config.setdefault, ["nan_fill", "nan_ratio"], ["past", 0.])
        self._id2valid_indices = [
            np.array([], np.int64) if len(indices) < self._num_history else
            np.arange(cumsum, cumsum + len(indices) - self._num_history + 1).astype(np.int64)
            for cumsum, indices in zip(self._num_samples_per_id_cumsum, self._id2indices)
        ]
        self._get_valid_samples_info()
        x, y = self.data.processed.xy
        feature_dim = self.data.processed_dim
        for i, valid_indices in enumerate(self._id2valid_indices):
            cumsum = self._num_valid_samples_per_id_cumsum[i]
            aggregated_flat_indices = self.aggregate(np.arange(cumsum, cumsum + len(valid_indices)))
            aggregated_x = x[aggregated_flat_indices].reshape([-1, self.num_aggregation, feature_dim])
            aggregated_x_nan_mask = np.isnan(aggregated_x)
            if y is None:
                aggregated_y_valid_mask = None
            else:
                aggregated_y = y[self.get_last_indices(aggregated_flat_indices)]
                aggregated_y_valid_mask = ~np.isnan(aggregated_y)
            aggregated_nan_ratio = aggregated_x_nan_mask.mean((1, 2))
            valid_mask = aggregated_nan_ratio <= nan_ratio
            if aggregated_y_valid_mask is not None:
                valid_mask &= aggregated_y_valid_mask.ravel()
            new_valid_indices = valid_indices[valid_mask]
            self._id2valid_indices[i] = new_valid_indices

    def aggregate(self, indices: np.ndarray) -> np.ndarray:
        """
        We've maintained two groups of indices in `_initialize` method:
        * the 'original' indices, which points to indices of original dataset
        * the 'valid' indices, which is 'virtual' should points to indices of the 'original' indices

        So we need to translate sampler indices to 'valid' indices, add offsets to the 'valid' indices,
        and then fetch the 'original' indices to fetch the corresponding data
        * _aggregate_core method will add offsets for us

        Parameters
        ----------
        indices : np.ndarray, indices come from sampler

        Returns
        -------
        indices : np.ndarray, aggregated & flattened 'original' indices, it should be available to
        reshape to [ -1, self.n_aggregation ]

        """
        valid_indices = self._id2valid_indices_stack[indices]
        aggregated_valid_indices_mat = self._aggregate_core(valid_indices[..., None])
        aggregated = self._id2indices_stack[aggregated_valid_indices_mat.ravel()]
        return aggregated.reshape([-1, self.num_aggregation])

    def get_last_indices(self, aggregated_flat_indices: np.ndarray):
        aggregated_indices_mat = aggregated_flat_indices.reshape([-1, self.num_aggregation])
        return aggregated_indices_mat[..., -1]

    @classmethod
    def register(cls, name: str):
        global aggregation_dict
        return register_core(name, aggregation_dict)


@AggregationBase.register("continuous")
class ContinuousAggregation(AggregationBase):
    def _initialize(self):
        self._history_arange = np.arange(self._num_history)
        super()._initialize()

    @property
    def num_aggregation(self):
        return self._num_history

    def _aggregate_core(self, indices: np.ndarray) -> np.ndarray:
        return indices + self._history_arange


class ImbalancedSampler(LoggingMixin):
    """
    Util class which can sample imbalance dataset in a balanced way

    Parameters
    ----------
    data : TabularData, data which we want to sample from
    imbalance_threshold : float
    * for binary class cases, if n_pos / n_neg < threshold, we'll treat data as imbalance data
    * for multi  class cases, if n_min_class / n_max_class < threshold, we'll treat data as imbalance data
    shuffle : bool, whether shuffle the returned indices
    sample_method : str, sampling method used in `cftool.misc.Sampler`
    * currently only 'multinomial' is supported
    verbose_level : int, verbose level used in `LoggingMixin`

    Examples
    ----------
    >>> import numpy as np
    >>>
    >>> from cfdata.types import np_int_type
    >>> from cfdata.tabular import TabularData
    >>> from cfdata.tabular.utils import ImbalancedSampler
    >>> from cftool.misc import get_counter_from_arr
    >>>
    >>> n = 20
    >>> x = np.arange(2 * n).reshape([n, 2])
    >>> # create an imbalance dataset
    >>> y = np.zeros([n, 1], np_int_type)
    >>> y[-1] = [1]
    >>> data = TabularData().read(x, y)
    >>> sampler = ImbalancedSampler(data)
    >>> # Counter({1: 12, 0: 8})
    >>> # This may vary, but will be rather balanced
    >>> # You might notice that positive samples are even more than negative samples!
    >>> print(get_counter_from_arr(y[sampler.get_indices()]))

    """

    def __init__(self,
                 data,
                 imbalance_threshold: float = 0.1,
                 *,
                 shuffle: bool = True,
                 aggregation: str = "continuous",
                 aggregation_config: Dict[str, Any] = None,
                 sample_method: str = "multinomial",
                 verbose_imbalance: bool = True,
                 verbose_level: int = 2):
        self.data = data
        self.shuffle = shuffle
        self.imbalance_threshold = imbalance_threshold
        self._sample_imbalance_flag = True
        if not data.is_ts:
            self.aggregation = None
            self._num_samples = len(data)
        else:
            if aggregation_config is None:
                aggregation_config = {}
            self.aggregation = aggregation_dict[aggregation](data, aggregation_config, data._verbose_level)
            self._num_samples = len(self.aggregation.indices2id)
        label_recognizer = data.recognizers[-1]
        if not self.shuffle or data.is_reg:
            label_counts = self._label_ratios = self._sampler = None
        else:
            label_counter = label_recognizer.counter
            transform_dict = label_recognizer.transform_dict
            label_counter = {transform_dict[k]: v for k, v in label_counter.items()}
            label_counts = np.array([label_counter[k] for k in sorted(label_counter)], np_float_type)
            self._label_ratios, max_label_count = label_counts / self._num_samples, label_counts.max()
            if label_counts.min() / max_label_count >= imbalance_threshold:
                self._sampler = None
            else:
                labels = data.processed.y.ravel()
                sample_weights = np.zeros(self._num_samples, np_float_type)
                for i, count in enumerate(label_counts):
                    sample_weights[labels == i] = max_label_count / count
                sample_weights /= sample_weights.sum()
                self._sampler = Sampler(sample_method, sample_weights)

        self._sample_method, self._verbose_level = sample_method, verbose_level
        if verbose_imbalance and self._sampler is not None:
            self.log_msg(
                f"using imbalanced sampler with label counts = {label_counts.tolist()}",
                self.info_prefix, 2
            )

    def __len__(self):
        return self._num_samples

    @property
    def is_imbalance(self) -> bool:
        return self._sampler is not None

    @property
    def sample_imbalance(self) -> bool:
        return self._sample_imbalance_flag

    @property
    def label_ratios(self) -> Union[None, np.ndarray]:
        return self._label_ratios

    def switch_imbalance_status(self, flag: bool) -> None:
        self._sample_imbalance_flag = flag

    def get_indices(self) -> np.ndarray:
        if not self.shuffle or not self._sample_imbalance_flag or not self.is_imbalance:
            indices = np.arange(self._num_samples).astype(np.int64)
        else:
            indices = self._sampler.sample(self._num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        if self.aggregation is not None:
            indices = self.aggregation.aggregate(indices)
        return indices

    def copy(self) -> "ImbalancedSampler":
        return ImbalancedSampler(
            self.data,
            self.imbalance_threshold,
            shuffle=self.shuffle,
            sample_method=self._sample_method,
            verbose_level=self._verbose_level,
            verbose_imbalance=False
        )


class LabelCollators:
    @staticmethod
    def reg_default(y_batch):
        assert len(y_batch) == 2
        return y_batch[1] - y_batch[0]

    @staticmethod
    def clf_default(y_batch):
        assert len(y_batch) == 2
        return y_batch[1] == y_batch[0]


class DataLoader:
    """
    Util class which can generated batches from `ImbalancedSampler`

    Examples
    ----------
    >>> import numpy as np
    >>>
    >>> from cfdata.types import np_int_type
    >>> from cfdata.tabular import TabularData
    >>> from cfdata.tabular.utils import DataLoader
    >>> from cfdata.tabular.utils import ImbalancedSampler
    >>> from cftool.misc import get_counter_from_arr
    >>>
    >>> n = 20
    >>> x = np.arange(2 * n).reshape([n, 2])
    >>> y = np.zeros([n, 1], np_int_type)
    >>> y[-1] = [1]
    >>> data = TabularData().read(x, y)
    >>> sampler = ImbalancedSampler(data)
    >>> loader = DataLoader(16, sampler)
    >>> y_batches = []
    >>> for x_batch, y_batch in loader:
    >>>     y_batches.append(y_batch)
    >>>     # (16, 1) (16, 1)
    >>>     # (4, 1) (4, 1)
    >>>     print(x_batch.shape, y_batch.shape)
    >>> # Counter({1: 11, 0: 9})
    >>> print(get_counter_from_arr(np.vstack(y_batches).ravel()))

    """

    def __init__(self,
                 batch_size: int,
                 sampler: ImbalancedSampler,
                 *,
                 num_siamese: int = 1,
                 return_indices: bool = False,
                 label_collator: callable = None,
                 verbose_level: int = 2):
        self._indices_in_use = None
        self._verbose_level = verbose_level
        self.data = sampler.data
        self.return_indices = return_indices
        if return_indices and num_siamese > 1:
            print(f"{LoggingMixin.warning_prefix}`return_indices` is set to False because siamese loader is used")
            self.return_indices = False
        self._num_siamese, self._label_collator = num_siamese, label_collator
        self._num_samples, self.sampler = len(self.data), sampler
        self.batch_size = min(self._num_samples, batch_size)
        self._verbose_level = verbose_level

    def __len__(self):
        n_iter = int(self._num_samples / self.batch_size)
        return n_iter + int(n_iter * self.batch_size < self._num_samples)

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        data_next = self._get_next_batch()
        if self._num_siamese == 1:
            return data_next
        all_data = [data_next] if self._check_full_batch(data_next) else []
        while len(all_data) < self._num_siamese:
            data_next = self._get_next_batch()
            if self._check_full_batch(data_next):
                all_data.append(data_next)
        x_batch, y_batch = zip(*all_data)
        if self._label_collator is not None:
            y_batch = self._label_collator(y_batch)
        return x_batch, y_batch

    @property
    def enabled_sampling(self) -> bool:
        return self.sampler.sample_imbalance

    @enabled_sampling.setter
    def enabled_sampling(self, value: bool):
        self.sampler.switch_imbalance_status(value)

    def _reset(self):
        reset_caches = {
            "_indices_in_use": self.sampler.get_indices(),
            "_siamese_cursor": 0, "_cursor": -1
        }
        for attr, init_value in reset_caches.items():
            setattr(self, attr, init_value)

    def _get_next_batch(self):
        n_iter, self._cursor = len(self), self._cursor + 1
        if self._cursor == n_iter * self._num_siamese:
            raise StopIteration
        if self._num_siamese > 1:
            new_siamese_cursor = int(self._cursor / n_iter)
            if new_siamese_cursor > self._siamese_cursor:
                self._siamese_cursor = new_siamese_cursor
                self._indices_in_use = self.sampler.get_indices()
        start = (self._cursor - n_iter * self._siamese_cursor) * self.batch_size
        end = start + self.batch_size
        indices = self._indices_in_use[start:end]
        batch = self.data[indices]
        if not self.return_indices:
            return batch
        return batch, indices

    def _check_full_batch(self, batch):
        if len(batch[0]) == self.batch_size:
            return True
        return False

    def copy(self) -> "DataLoader":
        return DataLoader(
            self.batch_size,
            self.sampler.copy(),
            num_siamese=self._num_siamese,
            return_indices=self.return_indices,
            label_collator=self._label_collator,
            verbose_level=self._verbose_level
        )


__all__ = [
    "split_file",
    "SplitResult", "TimeSeriesConfig", "DataSplitter", "KFold", "KRandom", "KBootstrap",
    "ImbalancedSampler", "LabelCollators", "DataLoader"
]
