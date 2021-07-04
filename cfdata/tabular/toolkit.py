import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.misc import get_unique_indices
from cftool.misc import Sampler
from cftool.misc import LoggingMixin

from abc import *
from .misc import *
from ..types import *
from .api import TabularData


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
    >>> from cfdata.tabular.api import TabularDataset
    >>> from cfdata.tabular.toolkit import KFold
    >>>
    >>> x = np.arange(12).reshape([6, 2])
    >>> # create an imbalance dataset
    >>> y = np.zeros(6, np_int_type)
    >>> y[[-1, -2]] = 1
    >>> dataset = TabularDataset.from_xy(x, y, "clf")
    >>> k_fold = KFold(3, dataset)
    >>> for train_fold, test_fold in k_fold:
    >>>     print(np.vstack([train_fold.dataset.x, test_fold.dataset.x]))
    >>>     print(np.vstack([train_fold.dataset.y, test_fold.dataset.y]))

    """

    def __init__(self, k: int, dataset: TabularDataset, **kwargs: Any):
        if k <= 1:
            raise ValueError("k should be larger than 1 in KFold")
        ratio = 1.0 / k
        self.n_list = (k - 1) * [ratio]
        self.splitter = DataSplitter(**kwargs).fit(dataset)
        self._cursor: int
        self._order: np.ndarray
        self.split_results: List[SplitResult]

    def __iter__(self) -> "KFold":
        self.split_results = self.splitter.split_multiple(
            self.n_list,
            return_remained=True,
        )
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
    >>> from cfdata.tabular.api import TabularDataset
    >>> from cfdata.tabular.toolkit import KRandom
    >>>
    >>> x = np.arange(12).reshape([6, 2])
    >>> # create an imbalance dataset
    >>> y = np.zeros(6, np_int_type)
    >>> y[[-1, -2]] = 1
    >>> dataset = TabularDataset.from_xy(x, y, "clf")
    >>> k_random = KRandom(3, 2, dataset)
    >>> for train_fold, test_fold in k_random:
    >>>     print(np.vstack([train_fold.dataset.x, test_fold.dataset.x]))
    >>>     print(np.vstack([train_fold.dataset.y, test_fold.dataset.y]))

    """

    def __init__(
        self,
        k: int,
        num_test: Union[int, float],
        dataset: TabularDataset,
        **kwargs: Any,
    ):
        self._cursor: int
        self.k, self.num_test = k, num_test
        self.splitter = DataSplitter(**kwargs).fit(dataset)

    def __iter__(self) -> "KRandom":
        self._cursor = 0
        return self

    def __next__(self) -> Tuple[SplitResult, SplitResult]:
        if self._cursor >= self.k:
            raise StopIteration
        self._cursor += 1
        self.splitter.reset()
        test_result, train_result = self.splitter.split_multiple(
            [self.num_test],
            return_remained=True,
        )
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
    >>> from cfdata.tabular.api import TabularDataset
    >>> from cfdata.tabular.toolkit import KBootstrap
    >>>
    >>> x = np.arange(12).reshape([6, 2])
    >>> # create an imbalance dataset
    >>> y = np.zeros(6, np_int_type)
    >>> y[[-1, -2]] = 1
    >>> dataset = TabularDataset.from_xy(x, y, "clf")
    >>> k_bootstrap = KBootstrap(3, 2, dataset)
    >>> for train_fold, test_fold in k_bootstrap:
    >>>     print(np.vstack([train_fold.dataset.x, test_fold.dataset.x]))
    >>>     print(np.vstack([train_fold.dataset.y, test_fold.dataset.y]))

    """

    def __init__(
        self,
        k: int,
        num_test: Union[int, float],
        dataset: TabularDataset,
        **kwargs: Any,
    ):
        self._cursor: int
        self.dataset = dataset
        self.num_samples = len(dataset)
        if isinstance(num_test, float):
            num_test = int(round(num_test * self.num_samples))
        self.k, self.num_test = k, num_test
        self.splitter = DataSplitter(**kwargs).fit(dataset)

    def __iter__(self) -> "KBootstrap":
        self._cursor = 0
        return self

    def __next__(self) -> Tuple[SplitResult, SplitResult]:
        if self._cursor >= self.k:
            raise StopIteration
        self._cursor += 1
        self.splitter.reset()
        test_result, train_result = self.splitter.split_multiple(
            [self.num_test],
            return_remained=True,
        )
        tr_indices = train_result.corresponding_indices
        tr_indices = np.random.choice(tr_indices, len(tr_indices))
        tr_set = self.dataset.split_with(tr_indices)
        tr_split = SplitResult(tr_set, tr_indices, None)
        return tr_split, test_result


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
    >>> from cfdata.tabular.toolkit import ImbalancedSampler
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

    def __init__(
        self,
        data: TabularData,
        imbalance_threshold: float = 0.1,
        *,
        shuffle: bool = True,
        aggregation: str = "continuous",
        aggregation_config: Optional[Dict[str, Any]] = None,
        sample_weights: Optional[np.ndarray] = None,
        sample_method: str = "multinomial",
        verbose_imbalance: bool = True,
        verbose_level: int = 2,
    ):
        self.data = data
        self.shuffle = shuffle
        self.imbalance_threshold = imbalance_threshold
        self._sample_imbalance_flag = True
        self._aggregation_name = aggregation
        self._aggregation_config = aggregation_config
        if not data.is_ts:
            self.aggregation = None
            self._num_samples = len(data)
        else:
            if aggregation_config is None:
                aggregation_config = {}
            base = aggregation_dict[aggregation]
            self.aggregation = base(data, aggregation_config, data._verbose_level)
            self._num_samples = len(self.aggregation.indices2id)
        if sample_weights is not None:
            label_counts = None
            self.sample_weights = sample_weights.copy()
            self.sample_weights /= self.sample_weights.sum() + 1e-8
            self._sampler = Sampler(sample_method, self.sample_weights)
        else:
            self.sample_weights = None
            if not self.shuffle or data.is_reg:
                label_counts = self._label_ratios = self._sampler = None
            else:
                label_recognizer = data.recognizers[-1]
                if label_recognizer is None:
                    raise ValueError(
                        "`data` should contain label recognizer "
                        "for `ImbalancedSampler`"
                    )
                label_counter = label_recognizer.counter
                transform_dict = label_recognizer.transform_dict
                new_counter = {transform_dict[k]: v for k, v in label_counter.items()}
                counts_list = [new_counter[k] for k in sorted(new_counter)]
                label_counts = np.array(counts_list, np_float_type)
                self._label_ratios = label_counts / self._num_samples
                max_label_count = label_counts.max()
                if label_counts.min() / max_label_count >= imbalance_threshold:
                    self._sampler = None
                else:
                    processed = data.processed
                    if processed is None:
                        raise ValueError(
                            "`data` should contain `processed` "
                            "for `ImbalancedSampler`"
                        )
                    if not isinstance(processed.y, np.ndarray):
                        raise ValueError(
                            "`data` should contain `processed.y` "
                            "for `ImbalancedSampler`"
                        )
                    labels = processed.y.ravel()
                    sample_weights = np.zeros(self._num_samples, np_float_type)
                    for i, count in enumerate(label_counts):
                        sample_weights[labels == i] = max_label_count / count
                    sample_weights /= sample_weights.sum() + 1e-8
                    self._sampler = Sampler(sample_method, sample_weights)

        self._sample_method = sample_method
        self._verbose_level = verbose_level
        if label_counts is not None and verbose_imbalance:
            if self._sampler is not None:
                self.log_msg(
                    "using imbalanced sampler with "
                    f"label counts = {label_counts.tolist()}",
                    self.info_prefix,
                    2,
                )

    def __len__(self) -> int:
        return self._num_samples

    @property
    def is_imbalance(self) -> bool:
        return self._sampler is not None

    @property
    def sample_imbalance(self) -> bool:
        return self._sample_imbalance_flag

    @property
    def label_ratios(self) -> Optional[np.ndarray]:
        return self._label_ratios

    def switch_imbalance_status(self, flag: bool) -> None:
        self._sample_imbalance_flag = flag

    def get_indices(self) -> np.ndarray:
        if not self.shuffle or not self._sample_imbalance_flag or not self.is_imbalance:
            indices = np.arange(self._num_samples).astype(np_int_type)
        else:
            if self._sampler is None:
                raise ValueError("`_sampler` is not yet generated")
            indices = self._sampler.sample(self._num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        if self.aggregation is not None:
            indices = self.aggregation.aggregate(indices)
        return indices

    def copy(self) -> "ImbalancedSampler":
        aggregation_config = None
        if self._aggregation_config is not None:
            aggregation_config = shallow_copy_dict(self._aggregation_config)
        sample_weights = None
        if self.sample_weights is not None:
            sample_weights = self.sample_weights.copy()
        return ImbalancedSampler(
            self.data,
            self.imbalance_threshold,
            shuffle=self.shuffle,
            aggregation=self._aggregation_name,
            aggregation_config=aggregation_config,
            sample_weights=sample_weights,
            sample_method=self._sample_method,
            verbose_level=self._verbose_level,
            verbose_imbalance=False,
        )


class DataLoader:
    """
    Util class which can generated batches from `ImbalancedSampler`

    Examples
    ----------
    >>> import numpy as np
    >>>
    >>> from cfdata.types import np_int_type
    >>> from cfdata.tabular import TabularData
    >>> from cfdata.tabular.toolkit import DataLoader, ImbalancedSampler
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

    def __init__(
        self,
        batch_size: int,
        sampler: ImbalancedSampler,
        *,
        return_indices: bool = False,
        label_collator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        verbose_level: int = 2,
    ):
        self._cursor: int
        self._indices_in_use = None
        self._verbose_level = verbose_level
        self.data = sampler.data
        self.sampler = sampler
        self.return_indices = return_indices
        self._label_collator = label_collator
        self._num_samples = len(sampler)
        self.batch_size = min(self._num_samples, batch_size)

    def __len__(self) -> int:
        n_iter = int(self._num_samples / self.batch_size)
        return n_iter + int(n_iter * self.batch_size < self._num_samples)

    def __iter__(self) -> "DataLoader":
        self._reset()
        return self

    def __next__(self) -> batch_type:
        data_next = self._get_next_batch()
        if self.return_indices:
            (x_batch, y_batch), indices = data_next
        else:
            indices = None
            x_batch, y_batch = data_next
        if self._label_collator is not None:
            y_batch = self._label_collator(y_batch)
        batch = x_batch, y_batch
        if not self.return_indices:
            return batch
        return batch, indices

    @property
    def enabled_sampling(self) -> bool:
        return self.sampler.sample_imbalance

    @enabled_sampling.setter
    def enabled_sampling(self, value: bool) -> None:
        self.sampler.switch_imbalance_status(value)

    def _reset(self) -> None:
        self._cursor = -1
        self._indices_in_use = self.sampler.get_indices()

    def _get_next_batch(self) -> batch_type:
        n_iter, self._cursor = len(self), self._cursor + 1
        if self._cursor == n_iter:
            raise StopIteration
        if self._indices_in_use is None:
            raise ValueError("`_indices_in_use` is not yet generated")
        start = self._cursor * self.batch_size
        end = start + self.batch_size
        indices = self._indices_in_use[start:end]
        batch = self.data[indices]
        if not self.return_indices:
            return batch
        return batch, indices

    def _check_full_batch(self, data_item: data_item_type) -> bool:
        if len(data_item[0]) == self.batch_size:
            return True
        return False

    def copy(self) -> "DataLoader":
        return DataLoader(
            self.batch_size,
            self.sampler.copy(),
            return_indices=self.return_indices,
            label_collator=self._label_collator,
            verbose_level=self._verbose_level,
        )


# time series


aggregation_dict: Dict[str, Type["AggregationBase"]] = {}


class AggregationBase(LoggingMixin, metaclass=ABCMeta):
    def __init__(self, data: TabularData, config: Dict[str, Any], verbose_level: int):
        if not data.is_ts:
            raise ValueError("time series data is required")
        self.data = data
        self.config = config
        self._verbose_level = verbose_level
        self._num_history = config.setdefault("num_history", 1)
        raw = data.raw
        if raw is None:
            raise ValueError("`data` need to contain `raw` for `AggregationBase`")
        if raw.xT is None:
            raise ValueError("`data` need to contain `raw.xT` for `AggregationBase`")
        if data.ts_config is None:
            raise ValueError("`data` need to contain `ts_config` for `AggregationBase`")
        id_column_idx = data.ts_config.id_column_idx
        if id_column_idx is None:
            msg = "`ts_config` need to contain `id_column_idx` for `AggregationBase`"
            raise ValueError(msg)
        id_column = raw.xT[id_column_idx]
        sorted_id_column = [id_column[i] for i in data.ts_sorting_indices]
        unique_indices = get_unique_indices(np.array(sorted_id_column))
        self.indices2id: np.ndarray
        self._unique_id_arr = unique_indices.unique
        self._id2indices = unique_indices.split_indices
        self._initialize()

    @property
    @abstractmethod
    def num_aggregation(self) -> int:
        pass

    @abstractmethod
    def _aggregate_core(self, indices: np.ndarray) -> np.ndarray:
        """indices should be a column vector"""

    def _initialize(self) -> None:
        num_list = list(map(len, self._id2indices))
        self._num_samples_per_id = np.array(num_list, np_int_type)
        self.log_msg("generating valid aggregation info", self.info_prefix, 5)
        valid_mask = self._num_samples_per_id >= self._num_history
        if not valid_mask.any():
            raise ValueError(
                "current settings lead to empty valid dataset, "
                "increasing raw dataset size or decreasing n_history "
                f"(current: {self._num_history}) might help"
            )
        if not valid_mask.all():
            invalid_mask = ~valid_mask
            n_invalid_id = invalid_mask.sum()
            n_invalid_samples = self._num_samples_per_id[invalid_mask].sum()
            self.log_msg(
                f"{n_invalid_id} id (with {n_invalid_samples} samples) "
                f"will be dropped (n_history={self._num_history})",
                self.info_prefix,
                verbose_level=2,
            )
            invalid_ids = self._unique_id_arr[invalid_mask].tolist()
            self.log_msg(
                f"dropped id : {', '.join(map(str, invalid_ids))}",
                self.info_prefix,
                verbose_level=4,
            )
        self._num_samples_per_id_cumsum = np.hstack(
            [[0], np.cumsum(self._num_samples_per_id[:-1])]
        )
        # self._id2indices need to contain 'redundant' indices here because
        # aggregation need to aggregate those 'invalid' samples
        self._id2indices_stack = np.hstack(self._id2indices)
        self.log_msg(
            "generating aggregation attributes",
            self.info_prefix,
            verbose_level=5,
        )
        self._get_id2valid_indices()
        self._inject_valid_samples_info()

    def _inject_valid_samples_info(self) -> None:
        # 'indices' in self.indices2id here doesn't refer to indices of original dataset
        # (e.g. 'indices' in self._id2indices), but refers to indices generated by sampler,
        # so we should only care 'valid' indices here
        self._num_valid_samples_per_id = list(map(len, self._id2valid_indices))
        self._num_valid_samples_per_id_cumsum = np.hstack(
            [[0], np.cumsum(self._num_valid_samples_per_id[:-1])]
        )
        num_int = self._num_valid_samples_per_id_cumsum.astype(np_int_type)
        self._num_valid_samples_per_id_cumsum = num_int
        arange = np.arange(len(self._unique_id_arr))
        self.indices2id = np.repeat(arange, self._num_valid_samples_per_id)
        self._id2valid_indices_stack = np.hstack(self._id2valid_indices)

    def _get_id2valid_indices(self) -> None:
        # TODO : support nan_fill here
        nan_fill = self.config.setdefault("nan_fill", "past")
        nan_ratio = self.config.setdefault("nan_ratio", 0.0)
        self._id2valid_indices = [
            np.array([], np_int_type)
            if len(indices) < self._num_history
            else np.arange(
                cumsum, cumsum + len(indices) - self._num_history + 1
            ).astype(np_int_type)
            for cumsum, indices in zip(
                self._num_samples_per_id_cumsum, self._id2indices
            )
        ]
        self._inject_valid_samples_info()
        processed = self.data.processed
        if processed is None:
            raise ValueError("`processed` is not generated yet")
        x, y = processed.xy
        assert isinstance(x, np.ndarray)
        feature_dim = self.data.processed_dim
        for i, valid_indices in enumerate(self._id2valid_indices):
            cumsum = self._num_valid_samples_per_id_cumsum[i]
            arange = np.arange(cumsum, cumsum + len(valid_indices))
            aggregated_flat_indices = self.aggregate(arange).ravel()
            aggregated_x = x[aggregated_flat_indices]
            shape = [-1, self.num_aggregation, feature_dim]
            aggregated_x = aggregated_x.reshape(shape)
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
        * the 'valid' indices, which is 'virtual' should points to indices
        of the 'original' indices

        So we need to translate sampler indices to 'valid' indices, add offsets
        to the 'valid' indices, and then fetch the 'original' indices to fetch
        the corresponding data
        * _aggregate_core method will add offsets for us

        Parameters
        ----------
        indices : np.ndarray, indices come from sampler

        Returns
        -------
        indices : np.ndarray, aggregated 'original' indices

        """
        valid_indices = self._id2valid_indices_stack[indices]
        aggregated_valid_indices_mat = self._aggregate_core(valid_indices[..., None])
        aggregated = self._id2indices_stack[aggregated_valid_indices_mat.ravel()]
        reversed_aggregated = self.data.ts_sorting_indices[aggregated]
        return reversed_aggregated.reshape([-1, self.num_aggregation])

    def get_last_indices(self, aggregated_flat_indices: np.ndarray) -> np.ndarray:
        reshaped = aggregated_flat_indices.reshape([-1, self.num_aggregation])
        return reshaped[..., -1]

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global aggregation_dict
        return register_core(name, aggregation_dict)


@AggregationBase.register("continuous")
class ContinuousAggregation(AggregationBase):
    def _initialize(self) -> None:
        self._history_arange = np.arange(self._num_history)
        super()._initialize()

    @property
    def num_aggregation(self) -> int:
        return self._num_history

    def _aggregate_core(self, indices: np.ndarray) -> np.ndarray:
        return indices + self._history_arange


__all__ = [
    "KFold",
    "KRandom",
    "KBootstrap",
    "ImbalancedSampler",
    "DataLoader",
    "aggregation_dict",
    "AggregationBase",
]
