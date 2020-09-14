import numpy as np

from typing import *
from cftool.misc import *

from .misc import *
from ..types import *
from .wrapper import TabularData


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
    >>> from cfdata.tabular.misc import TaskTypes
    >>> from cfdata.tabular.toolkit import KFold
    >>> from cfdata.tabular.wrapper import TabularDataset
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
    >>> from cfdata.tabular.misc import TaskTypes
    >>> from cfdata.tabular.toolkit import KRandom
    >>> from cfdata.tabular.wrapper import TabularDataset
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
    >>> from cfdata.tabular.misc import TaskTypes
    >>> from cfdata.tabular.toolkit import KBootstrap
    >>> from cfdata.tabular.wrapper import TabularDataset
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

    def __init__(self,
                 data: TabularData,
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
        self._aggregation_name = aggregation
        self._aggregation_config = aggregation_config
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
        aggregation_config = None
        if self._aggregation_config is not None:
            aggregation_config = shallow_copy_dict(self._aggregation_config)
        return ImbalancedSampler(
            self.data,
            self.imbalance_threshold,
            shuffle=self.shuffle,
            aggregation=self._aggregation_name,
            aggregation_config=aggregation_config,
            sample_method=self._sample_method,
            verbose_level=self._verbose_level,
            verbose_imbalance=False,
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
        self._num_samples, self.sampler = len(sampler), sampler
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
            verbose_level=self._verbose_level,
        )


__all__ = [
    "KFold", "KRandom", "KBootstrap",
    "ImbalancedSampler", "LabelCollators", "DataLoader",
]
