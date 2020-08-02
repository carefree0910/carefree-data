import numpy as np

from ...types import *
from ....types import *
from ..base import Converter


@Converter.register("string")
class StringConverter(Converter):
    def _fit(self) -> "Converter":
        assert self.info.is_valid and self.info.is_string and self.info.need_transform
        self._transform_dict = self._recognizer.transform_dict
        self._reverse_transform_dict = dict(map(reversed, self._transform_dict.items()))
        return self

    def _convert(self,
                 flat_arr: flat_arr_type) -> np.ndarray:
        return np.array([self._transform_dict.get(elem, 0) for elem in flat_arr], np_float_type)

    def _recover(self,
                 flat_arr: flat_arr_type) -> np.ndarray:
        return np.array([self._reverse_transform_dict[elem] for elem in flat_arr], np.str)


__all__ = ["StringConverter"]
