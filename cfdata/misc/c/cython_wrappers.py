from typing import Any
from typing import Dict
from typing import Union

try:
    from .cython_utils import *
except ImportError:
    raise


import numpy as np


def _to_list(array: Union[list, np.ndarray]) -> list:
    if isinstance(array, np.ndarray):
        array = array.tolist()
    return array


def c_transform_flat_data_with_dict(
    flat_data: np.ndarray,
    transform_dict: Dict[Any, Any],
    need_truncate: bool,
) -> np.ndarray:
    nan_value = transform_dict.get("nan", 0)
    oob_value = nan_value if need_truncate else 0
    return transform_flat_data_with_dict(  # type: ignore
        flat_data,
        nan_value,
        oob_value,
        transform_dict,
    )


def c_is_all_numeric(array: np.ndarray) -> bool:
    return is_all_numeric(_to_list(array))  # type: ignore


def c_flat_arr_to_float32(array: np.ndarray) -> np.ndarray:
    return flat_arr_to_float32(_to_list(array))  # type: ignore


__all__ = [
    "c_transform_flat_data_with_dict",
    "c_is_all_numeric",
    "c_flat_arr_to_float32",
]
