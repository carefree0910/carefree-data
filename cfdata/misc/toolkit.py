import dill
import time
import hashlib
import datetime
import operator
import unicodedata

import numpy as np
import matplotlib.pyplot as plt

from functools import reduce
from typing import *

dill._dill._reverse_typemap["ClassType"] = type


# util functions

def timestamp(simplify=False, ensure_different=False):
    """
    Return current timestamp

    Parameters
    ----------
    simplify : bool. If True, format will be simplified to 'year-month-day'
    ensure_different : bool. If True, format will include millisecond

    Returns
    -------
    timestamp : str

    """

    now = datetime.datetime.now()
    if simplify:
        return now.strftime("%Y-%m-%d")
    if ensure_different:
        return now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def prod(iterable):
    """ return cumulative production of an iterable """

    return float(reduce(operator.mul, iterable, 1))


def hash_code(code, encode=True):
    """ return hash code for a string """

    if encode:
        code = code.encode()
    return hashlib.md5(code).hexdigest()[:8]


def prefix_dict(d, prefix):
    """ prefix every key in dict `d` with `prefix` """

    return {f"{prefix}_{k}": v for k, v in d.items()}


def check_params(module):
    """
    Check out whether the param definitions in module is correct

    Parameters
    ----------
    module : torch.nn.Module
        Should be a torch module with `main_params` & `aux_params` defined

    """

    assert not (set(module.main_params) & set(module.aux_params))
    assert set(module.parameters()) == set(module.main_params) | set(module.aux_params)
    module._params_checked_ = True


def shallow_copy_dict(d: dict) -> dict:
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = shallow_copy_dict(v)
    return d


def update_dict(src_dict: dict, tgt_dict: dict) -> dict:
    """
    Update tgt_dict with src_dict
    * Notice that changes will happen only on keys which src_dict holds

    Parameters
    ----------
    src_dict : dict
    tgt_dict : dict

    Returns
    -------
    tgt_dict : dict

    """

    for k, v in src_dict.items():
        tgt_v = tgt_dict.get(k)
        if tgt_v is None:
            tgt_dict[k] = v
        elif not isinstance(v, dict):
            tgt_dict[k] = v
        else:
            update_dict(v, tgt_v)
    return tgt_dict


def fix_float_to_length(num: float, length: int) -> str:
    """ change a float number to string format with fixed length """

    str_num = f"{num:f}"
    if str_num == "nan":
        return f"{str_num:^{length}s}"
    length = max(length, str_num.find("."))
    return str_num[:length].ljust(length, "0")


def truncate_string_to_length(string: str, length: int) -> str:
    """ truncate a string to make sure its length not exceeding a given length """

    if len(string) <= length:
        return string
    half_length = int(0.5 * length) - 1
    return string[:half_length] + "." * (length - 2 * half_length) + string[-half_length:]


def grouped(iterable: Iterable, n: int, *, keep_tail=False) -> List[tuple]:
    """ group an iterable every `n` elements """

    if not keep_tail:
        return list(zip(*[iter(iterable)] * n))
    with general_batch_manager(iterable, batch_size=n) as manager:
        return [tuple(batch) for batch in manager]


def is_numeric(s: str) -> bool:
    """ check whether `s` is a number """

    try:
        s = float(s)
        return True
    except (TypeError, ValueError):
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            return False


def get_one_hot(feature, num):
    """
    Get one-hot representation

    Parameters
    ----------
    feature : array-like, source data of one-hot representation
    num : int, dimension of the one-hot representation

    Returns
    -------
    one_hot : np.ndarray, one-hot representation of `feature`

    """

    if feature is None:
        return
    one_hot = np.zeros([len(feature), num], np.int64)
    one_hot[range(len(one_hot)), np.asarray(feature, np.int64).ravel()] = 1
    return one_hot


def show_or_save(export_path, fig=None, **kwargs):
    """
    Utility function to deal with figure

    Parameters
    ----------
    export_path : {None, str}
        * None : the figure will be shown
        * str  : it represents the path where the figure should be saved to
    fig : {None, plt.Figure}
        * None       : default figure contained in plt will be executed
        * plt.Figure : it will be executed

    """

    if export_path is None:
        fig.show(**kwargs) if fig is not None else plt.show(**kwargs)
    else:
        fig.savefig(export_path) if fig is not None else plt.savefig(export_path, **kwargs)
    plt.close()


def get_indices_from_another(base, segment):
    """
    Get `segment` elements' indices in `base`

    Warnings
    ----------
    All elements in segment should appear in base to ensure validity

    Parameters
    ----------
    base : np.ndarray, base array
    segment : np.ndarray, segment array

    Returns
    -------
    indices : np.ndarray, positions where elements in `segment` appear in `base`

    Examples
    -------
    >>> import numpy as np
    >>> base, segment = np.arange(100), np.random.permutation(100)[:10]
    >>> assert np.allclose(get_indices_from_another(base, segment), segment)

    """
    base_sorted_args = np.argsort(base)
    positions = np.searchsorted(base[base_sorted_args], segment)
    return base_sorted_args[positions]


def get_unique_indices(arr, return_raw=False):
    """
    Get indices for unique values of an array

    Parameters
    ----------
    arr : np.ndarray, target array which we wish to find indices of each unique value
    return_raw : bool, whether returning raw information

    Returns
    -------
    unique : np.ndarray, unique values of the given array (`arr`)
    unique_cnt : np.ndarray, counts of each unique value
        * If `return_raw`:
            sorting_indices : np.ndarray, indices which can (stably) sort the given array by its value
            split_arr : np.ndarray, array which can split the `sorting_indices` to make sure that each portion
            of the split indices belong & only belong to one of the unique values
        * If not `return_raw`:
            split_indices : list[np.ndarray], list of indices, each indices belong & only belong to
                one of the unique values

    Examples
    -------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 2, 4, 1, 0, 1], np.int64)
    >>> print(get_unique_indices(arr, return_raw=True), get_unique_indices(arr)[-1])
    >>> # [0, 1, 2, 3, 4]
    >>> # [1, 3, 2, 1, 1]
    >>> # [6, 0, 5, 7, 1, 3, 2, 4]
    >>> # [1, 4, 6, 7]
    >>> # [ [6], [0, 5, 7], [1, 3], [2], [4] ]

    """
    unique, unique_inv, unique_cnt = np.unique(arr, return_inverse=True, return_counts=True)
    sorting_indices, split_arr = np.argsort(unique_inv, kind="mergesort"), np.cumsum(unique_cnt)[:-1]
    if return_raw:
        return unique, unique_cnt, sorting_indices, split_arr
    return unique, unique_cnt, np.split(sorting_indices, split_arr)


def get_counter_from_arr(arr):
    if isinstance(arr, np.ndarray):
        arr = dict(zip(*np.unique(arr, return_counts=True)))
    return Counter(arr)


def register_core(name: str,
                  global_dict: Dict[str, type], *,
                  before_register: callable = None,
                  after_register: callable = None):
    def _register(cls):
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None:
            print(f"~~~ [warning] '{name}' has already registered "
                  f"in the given global dict ({global_dict})")
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls
    return _register


# constants

INFO_PREFIX = "~~~  [ info ] "


# contexts

class context_error_handler:
    """ Util class which provides exception handling when using context manager """

    @property
    def exception_suffix(self):
        return ""

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        pass

    def _exception_exit(self, exc_type, exc_val, exc_tb):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self._normal_exit(exc_type, exc_val, exc_tb)
        else:
            self._exception_exit(exc_type, exc_val, exc_tb)


class timeit(context_error_handler):
    """
    Timing context manager

    Examples
    --------
    >>> with timeit("something"):
    >>>     # do something here
    >>> # will print "~~~  [ info ] timing for    something     : x.xxxx"

    """

    def __init__(self, msg, precision=6):
        self._msg = msg
        self._p = precision

    def __enter__(self):
        self._t = time.time()

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        print(f"{INFO_PREFIX}timing for {self._msg:^16s} : {time.time() - self._t:{self._p}.{self._p-2}f}")


class general_batch_manager(context_error_handler):
    """
    Inference in batch, it could be any general instance

    Parameters
    ----------
    inputs : tuple(np.ndarray), auxiliary array inputs.
    n_elem : {int, float}, indicates how many elements will be processed in a batch
    batch_size : int, indicates the batch_size; if None, batch_size will be calculated by n_elem

    Examples
    --------
    >>> instance = type("test", (object,), {})()
    >>> with general_batch_manager(instance, np.arange(5), np.arange(1, 6), batch_size=2) as manager:
    >>>     for arr, tensor in manager:
    >>>         print(arr, tensor)
    >>>         # Will print:
    >>>         #   [0 1], [1 2]
    >>>         #   [2 3], [3 4]
    >>>         #   [4]  , [5]

    """

    def __init__(self, *inputs, n_elem=1e6, batch_size=None, max_batch_size=1024):
        if not inputs:
            raise ValueError("inputs should be provided in general_batch_manager")
        input_lengths = list(map(len, inputs))
        self._n, self._rs, self._inputs = input_lengths[0], [], inputs
        assert all(length == self._n for length in input_lengths), "inputs should be of same length"
        if batch_size is not None:
            self._batch_size = batch_size
        else:
            n_elem = int(n_elem)
            self._batch_size = int(n_elem / sum(map(lambda arr: prod(arr.shape[1:]), inputs)))
        self._batch_size = min(max_batch_size, min(self._n, self._batch_size))
        self._n_epoch = int(self._n / self._batch_size)
        self._n_epoch += int(self._n_epoch * self._batch_size < self._n)

    def __enter__(self):
        return self

    def __iter__(self):
        self._start, self._end = 0, self._batch_size
        return self

    def __next__(self):
        if self._start >= self._n:
            raise StopIteration
        batched_data = tuple(map(lambda arr: arr[self._start:self._end], self._inputs))
        self._start, self._end = self._end, self._end + self._batch_size
        if len(batched_data) == 1:
            return batched_data[0]
        return batched_data

    def __len__(self):
        return self._n_epoch


__all__ = [
    "get_indices_from_another", "get_unique_indices", "get_counter_from_arr", "get_one_hot", "hash_code",
    "prefix_dict", "check_params", "timestamp", "fix_float_to_length", "truncate_string_to_length", "grouped",
    "is_numeric", "show_or_save", "update_dict", "context_error_handler", "timeit",
    "general_batch_manager", "prod", "shallow_copy_dict", "register_core"
]
