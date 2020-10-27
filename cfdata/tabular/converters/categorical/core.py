import numpy as np

from ...misc import *
from ....types import *
from ..base import Converter
from ....misc.c import transform_flat_data_with_dict


@Converter.register("categorical")
class CategoricalConverter(Converter):
    def _fit(self) -> "Converter":
        assert self.info.is_valid and self.info.is_categorical
        self._transform_dict = self._recognizer.transform_dict
        self._reverse_transform_dict = {v: k for k, v in self._transform_dict.items()}
        return self

    def _convert(self, flat_arr: flat_arr_type) -> np.ndarray:
        flat_arr = np.asarray(flat_arr, np_float_type)
        if not self.info.need_transform:
            return flat_arr.astype(np_int_type)
        return transform_flat_data_with_dict(
            flat_arr,
            self._transform_dict,
            self.info.need_truncate,
        )

    def _recover(self, flat_arr: flat_arr_type) -> np.ndarray:
        flat_arr = np.asarray(flat_arr, np_float_type)
        if not self.info.need_transform:
            return flat_arr.astype(np_float_type)
        return transform_flat_data_with_dict(
            flat_arr,
            self._reverse_transform_dict,
            self.info.need_truncate,
        )


__all__ = ["CategoricalConverter"]
