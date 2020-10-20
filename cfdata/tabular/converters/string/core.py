import numpy as np

from ...misc import *
from ....types import *
from ..base import Converter


@Converter.register("string")
class StringConverter(Converter):
    def _fit(self) -> "Converter":
        assert self.info.is_valid and self.info.is_string and self.info.need_transform
        self._transform_dict = self._recognizer.transform_dict
        reversed_dict = {v: k for k, v in self._transform_dict.items()}
        self._reverse_transform_dict = reversed_dict
        return self

    def _convert(self, flat_arr: flat_arr_type) -> np.ndarray:
        converted = [self._transform_dict.get(elem, 0) for elem in flat_arr]
        return np.array(converted, np_float_type)

    def _recover(self, flat_arr: flat_arr_type) -> np.ndarray:
        recovered = [self._reverse_transform_dict[elem] for elem in flat_arr]
        return np.array(recovered, np.str)


__all__ = ["StringConverter"]
