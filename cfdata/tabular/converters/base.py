import os

import numpy as np

from typing import *
from abc import ABC, abstractmethod
from cftool.misc import register_core
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.misc import SavingMixin

from ..misc import *
from ..recognizer import Recognizer

converter_dict: Dict[str, Type["Converter"]] = {}


class Converter(SavingMixin, ABC):
    def __init__(self,
                 recognizer: Recognizer,
                 *,
                 inplace: bool = False,
                 **kwargs):
        self._config = kwargs
        self._inplace = inplace
        self._recognizer = recognizer

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

    @property
    def cache_excludes(self):
        return {"_recognizer"}

    @property
    def data_tuple_base(self) -> Optional[Type[NamedTuple]]:
        return

    @property
    def data_tuple_attributes(self) -> Optional[List[str]]:
        return

    def _initialize(self, **kwargs) -> None:
        pass

    def initialize(self):
        self._initialize(**self._config)
        self._fit()

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

    identifier_file = "identifier.txt"
    recognizer_folder = "__recognizer"

    def save(self,
             folder: str,
             *,
             compress: bool = True,
             remove_original: bool = True):
        super().save(folder, compress=False)
        abs_folder = os.path.abspath(folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [folder]):
            with open(os.path.join(abs_folder, self.identifier_file), "w") as f:
                f.write(self.__identifier__)
            recognizer_folder = os.path.join(abs_folder, self.recognizer_folder)
            self._recognizer.save(
                recognizer_folder,
                compress=compress,
                remove_original=remove_original,
            )
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)

    @classmethod
    def load(cls,
             folder: str,
             *,
             compress: bool = True):
        abs_folder = os.path.abspath(folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [folder]):
            with Saving.compress_loader(
                folder,
                compress,
                remove_extracted=True,
            ):
                with open(os.path.join(abs_folder, cls.identifier_file), "r") as f:
                    identifier = f.read().strip()
                recognizer_folder = os.path.join(abs_folder, cls.recognizer_folder)
                recognizer = Recognizer.load(recognizer_folder, compress=compress)
                converter = converter_dict[identifier](recognizer)
                Saving.load_instance(converter, folder, log_method=converter.log_msg)
        return converter

    @classmethod
    def make_with(cls,
                  recognizer: Recognizer,
                  *,
                  inplace: bool = False,
                  **kwargs) -> "Converter":
        key = recognizer.info.column_type.value
        converter = converter_dict[key](recognizer, inplace=inplace, **kwargs)
        converter.initialize()
        return converter

    @classmethod
    def register(cls, name):
        global converter_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, converter_dict, before_register=before)


__all__ = ["Converter", "converter_dict"]
