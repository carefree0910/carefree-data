import numpy as np

from typing import *
from abc import ABC, abstractmethod

from ..misc import *
from ..recognizer import Recognizer
from cftool.misc import register_core

converter_dict: Dict[str, Type["Converter"]] = {}


class Converter(ABC):
    def __init__(self,
                 recognizer: Recognizer,
                 *,
                 inplace: bool = False,
                 **kwargs):
        self._inplace = inplace
        self._recognizer = recognizer
        self._initialize(**kwargs)
        self._fit()

    def __str__(self):
        return f"Converter({self.info.column_type})"

    __repr__ = __str__

    @abstractmethod
    def _fit(self) -> "Converter":
        pass

    @abstractmethod
    def _convert(self,
                 flat_arr: flat_arr_type) -> np.ndarray:
        pass

    @abstractmethod
    def _recover(self,
                 flat_arr: flat_arr_type) -> np.ndarray:
        pass

    @property
    def info(self):
        return self._recognizer.info

    @property
    def converted_input(self):
        if getattr(self, "_converted_features", None) is None:
            self._converted_features = self.convert(self.info.flat_arr)
        return self._converted_features

    def _initialize(self, **kwargs) -> None:
        pass

    def convert(self,
                flat_arr: flat_arr_type) -> np.ndarray:
        if not self._inplace:
            flat_arr = flat_arr.copy()
        return self._convert(flat_arr)

    def recover(self,
                flat_arr: flat_arr_type,
                *,
                inplace: bool = False) -> np.ndarray:
        if not inplace:
            flat_arr = flat_arr.copy()
        return self._recover(flat_arr)

    @classmethod
    def make_with(cls,
                  recognizer: Recognizer,
                  *,
                  inplace: bool = False,
                  **kwargs) -> "Converter":
        key = recognizer.info.column_type.value
        return converter_dict[key](recognizer, inplace=inplace, **kwargs)

    @classmethod
    def register(cls, name):
        global converter_dict
        return register_core(name, converter_dict)


__all__ = ["Converter", "converter_dict"]
