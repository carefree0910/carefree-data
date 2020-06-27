import numpy as np

from typing import *
from functools import partial
from cftool.misc import *

from .types import *
from ..types import *


class SplitResult(NamedTuple):
    dataset: TabularDataset
    corresponding_indices: np.ndarray
    remaining_indices: np.ndarray

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
                 time_series_config: dict = None,
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
            self._id_column_setting = time_series_config.get("id_column")
            self._time_column_setting = time_series_config.get("time_column")
            if self._id_column_setting is None:
                raise ValueError("id_column should be provided in time_series_config")
            if self._time_column_setting is None:
                raise ValueError("time_column should be provided in time_series_config")
            self._id_column_is_int, self._time_column_is_int = map(
                lambda column: isinstance(column, int), [self._id_column_setting, self._time_column_setting])

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
    def is_time_series(self):
        return self._time_series_config is not None

    @property
    def sorting_indices(self):
        if not self.is_time_series:
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
        n_data = len(self._x)
        if not self._shuffle:
            self._remained_indices = np.arange(n_data)
        else:
            self._remained_indices = np.random.permutation(n_data)
        self._remained_indices = self._remained_indices.astype(np_int_type)

    def _reset_clf(self):
        if self._label_indices_list is None:
            flattened_y = self._y.ravel()
            unique_indices = get_unique_indices(flattened_y)
            self._unique_labels, counts = unique_indices[:2]
            self._label_indices_list = unique_indices.split_indices
            self._n_samples = len(flattened_y)
            self._label_ratios = counts / self._n_samples
            self._n_unique_labels = len(self._unique_labels)
            if self._n_unique_labels == 1:
                raise ValueError("only 1 unique label is detected, which is invalid in classification task")
            self._unique_labels = self._unique_labels.astype(np_int_type)
            self._label_indices_list = list(map(partial(np.asarray, dtype=np_int_type), self._label_indices_list))
        self._reset_indices_list("label_indices_list")

    def _reset_time_series(self):
        if self._time_indices_list is None:
            self.log_msg(f"gathering time -> indices mapping", self.info_prefix, verbose_level=5)
            self._unique_times, times_counts, self._time_indices_list = map(
                lambda arr: arr[::-1],
                get_unique_indices(self._time_column)
            )
            self._times_counts_cumsum = np.cumsum(times_counts).astype(np_int_type)
            assert self._times_counts_cumsum[-1] == len(self._time_column)
            self._time_series_sorting_indices = np.hstack(self._time_indices_list[::-1]).astype(np_int_type)
            self._unique_times = self._unique_times.astype(np_int_type)
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
        if n < self._n_unique_labels:
            raise ValueError(
                f"at least {self._n_unique_labels} samples are required because "
                f"we have {self._n_unique_labels} unique labels"
            )
        pop_indices_list, tgt_indices_list = [], []
        n_samples_per_label = np.maximum(1, np.round(n * self._label_ratios).astype(np_int_type))
        # -n_unique_labels <= n_samples_exceeded <= n_unique_labels
        n_samples_exceeded = n_samples_per_label.sum() - n
        # adjust n_samples_per_label to make sure `n` samples are split out
        if n_samples_exceeded != 0:
            sign, n_samples_exceeded = np.sign(n_samples_exceeded), abs(n_samples_exceeded)
            chosen_indices = np.arange(self._n_unique_labels)[n_samples_per_label != 1]
            np.random.shuffle(chosen_indices)
            n_chosen_indices = len(chosen_indices)
            n_tile = int(np.ceil(n_samples_exceeded / n_chosen_indices))
            n_proceeded = 0
            for _ in range(n_tile - 1):
                n_samples_per_label[chosen_indices] -= sign
                n_proceeded += n_chosen_indices
            for idx in chosen_indices[:n_samples_exceeded - n_proceeded]:
                n_samples_per_label[idx] -= sign
        assert n_samples_per_label.sum() == n
        n_overlap = 0
        for indices, n_sample_per_label in zip(self._label_indices_list_in_use, n_samples_per_label):
            n_samples_in_use = len(indices)
            tgt_indices_list.append(indices[-n_sample_per_label:])
            if n_sample_per_label >= n_samples_in_use:
                pop_indices_list.append([])
                n_overlap += n_sample_per_label
            else:
                pop_indices_list.append(np.arange(n_samples_in_use - n_sample_per_label, n_samples_in_use))
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
            base = np.zeros(self._n_samples)
            base[tgt_indices] += 1
            base[remain_indices] += 1
            assert np.sum(base >= 2) <= n_overlap
            self._remained_indices = remain_indices
        return tgt_indices

    def _split_time_series(self, n: int):
        split_arg = np.argmax(self._times_counts_cumsum_in_use >= n)
        n_left = self._times_counts_cumsum_in_use[split_arg] - n
        if split_arg == 0:
            n_res, selected_indices = n, []
        else:
            n_res = n - self._times_counts_cumsum_in_use[split_arg - 1]
            selected_indices = self._time_indices_list_in_use[:split_arg]
            self._time_indices_list_in_use = self._time_indices_list_in_use[split_arg:]
            self._times_counts_cumsum_in_use = self._times_counts_cumsum_in_use[split_arg:]
        selected_indices.append(self._time_indices_list_in_use[0][:n_res])
        if n_left > 0:
            self._time_indices_list_in_use[0] = self._time_indices_list_in_use[0][n_res:]
        else:
            self._time_indices_list_in_use = self._time_indices_list_in_use[1:]
            self._times_counts_cumsum_in_use = self._times_counts_cumsum_in_use[1:]
        tgt_indices, self._remained_indices = map(np.hstack, [selected_indices, self._time_indices_list_in_use])
        self._times_counts_cumsum_in_use -= n
        return tgt_indices[::-1].copy()

    def fit(self,
            dataset: TabularDataset) -> "DataSplitter":
        self._dataset = dataset
        self._is_regression = dataset.is_reg
        self._x = dataset.x
        self._y = dataset.y
        if not self.is_time_series:
            self._time_column = None
        else:
            if not self._id_column_is_int and not self._time_column_is_int:
                self._id_column, self._time_column = map(
                    np.asarray, [self._id_column_setting, self._time_column_setting])
            else:
                id_column, time_column = self._id_column_setting, self._time_column_setting
                error_msg_prefix = "id_column & time_column should both be int, but"
                if not self._id_column_is_int:
                    raise ValueError(f"{error_msg_prefix} id_column='{id_column}' found")
                if not self._time_column_is_int:
                    raise ValueError(f"{error_msg_prefix} time_column='{time_column}' found")
                if id_column < time_column:
                    id_first = True
                    split_list = [id_column, id_column + 1, time_column, time_column + 1]
                else:
                    id_first = False
                    split_list = [time_column, time_column + 1, id_column, id_column + 1]
                columns = np.split(self._x, split_list, axis=1)
                if id_first:
                    self._id_column, self._time_column = columns[1], columns[3]
                else:
                    self._id_column, self._time_column = columns[3], columns[1]
                self._x = np.hstack([columns[0], columns[2], columns[4]])
            self._id_column, self._time_column = map(np.ravel, [self._id_column, self._time_column])
        return self.reset()

    def reset(self) -> "DataSplitter":
        if self._time_column is not None:
            self._reset_time_series()
        elif self._is_regression:
            self._reset_reg()
        else:
            self._reset_clf()
        return self

    def split(self,
              n: Union[int, float]) -> SplitResult:
        error_msg = "please call 'reset' method before calling 'split' method"
        if self._is_regression and self._remained_indices is None:
            raise ValueError(error_msg)
        if not self._is_regression and self._label_indices_list_in_use is None:
            raise ValueError(error_msg)
        if n >= len(self._remained_indices):
            remained_x, remained_y = self.remained_xy
            return SplitResult(
                TabularDataset.from_xy(remained_x, remained_y, self._dataset.task_type),
                self._remained_indices, np.array([], np.int)
            )
        if n < 1. or (n == 1. and isinstance(n, float)):
            n = int(round(len(self._x) * n))
        if self.is_time_series:
            split_method = self._split_time_series
        else:
            split_method = self._split_reg if self._is_regression else self._split_clf
        tgt_indices = split_method(n)
        assert len(tgt_indices) == n
        x_split, y_split = self._x[tgt_indices], self._y[tgt_indices]
        dataset_split = TabularDataset(x_split, y_split, *self._dataset[2:])
        return SplitResult(dataset_split, tgt_indices, self._remained_indices)

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
        if k <= 1:
            raise ValueError("k should be larger than 1 in KFold")
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
                 sample_method: str = "multinomial",
                 verbose_imbalance: bool = True,
                 verbose_level: int = 2):
        self.data = data
        self.imbalance_threshold = imbalance_threshold
        self._sample_imbalance_flag = True
        self.shuffle, self._n_samples = shuffle, len(data)
        if not self.shuffle or data.task_type is TaskTypes.REGRESSION:
            label_counts = self._label_ratios = self._sampler = None
        else:
            recognizer = data.recognizers[-1]
            label_counter = recognizer.counter
            transform_dict = recognizer.transform_dict
            label_counter = {transform_dict[k]: v for k, v in label_counter.items()}
            label_counts = np.array([label_counter[k] for k in sorted(label_counter)], np_float_type)
            self._label_ratios, max_label_count = label_counts / self._n_samples, label_counts.max()
            if label_counts.min() / max_label_count >= imbalance_threshold:
                self._sampler = None
            else:
                labels = data.processed.y.ravel()
                sample_weights = np.zeros(self._n_samples, np_float_type)
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
        return self._n_samples

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
            indices = np.arange(self._n_samples).astype(np.int64)
        else:
            indices = self._sampler.sample(self._n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
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
                 n_siamese: int = 1,
                 label_collator: callable = None,
                 verbose_level: int = 2):
        self._indices_in_use = None
        self._verbose_level = verbose_level
        self.data = sampler.data
        self._n_siamese, self._label_collator = n_siamese, label_collator
        self._n_samples, self.sampler = len(self.data), sampler
        self.batch_size = min(self._n_samples, batch_size)
        self._verbose_level = verbose_level

    def __len__(self):
        n_iter = int(self._n_samples / self.batch_size)
        return n_iter + int(n_iter * self.batch_size < self._n_samples)

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        data_next = self._get_next_batch()
        if self._n_siamese == 1:
            return data_next
        all_data = [data_next] if self._check_full_batch(data_next) else []
        while len(all_data) < self._n_siamese:
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
        if self._cursor == n_iter * self._n_siamese:
            raise StopIteration
        if self._n_siamese > 1:
            new_siamese_cursor = int(self._cursor / n_iter)
            if new_siamese_cursor > self._siamese_cursor:
                self._siamese_cursor = new_siamese_cursor
                self._indices_in_use = self.sampler.get_indices()
        start = (self._cursor - n_iter * self._siamese_cursor) * self.batch_size
        end = start + self.batch_size
        batch = self.data[self._indices_in_use[start:end]]
        return batch

    def _check_full_batch(self, batch):
        if len(batch[0]) == self.batch_size:
            return True
        return False

    def copy(self) -> "DataLoader":
        return DataLoader(
            self.batch_size,
            self.sampler.copy(),
            n_siamese=self._n_siamese,
            label_collator=self._label_collator,
            verbose_level=self._verbose_level
        )


__all__ = [
    "SplitResult", "DataSplitter", "KFold", "KRandom",
    "ImbalancedSampler", "LabelCollators", "DataLoader"
]
