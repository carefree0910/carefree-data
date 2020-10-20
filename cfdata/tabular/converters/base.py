import numpy as np

from typing import *
from abc import ABCMeta, abstractmethod
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict

from ..misc import *
from ..recognizer import Recognizer

converter_dict: Dict[str, Type["Converter"]] = {}


class Converter(DataStructure, metaclass=ABCMeta):
    def __init__(
        self,
        recognizer: Recognizer,
        *,
        inplace: bool = False,
        **kwargs: Any,
    ):
        self._config = kwargs
        self._inplace = inplace
        self._recognizer = recognizer

    def __str__(self) -> str:
        return f"Converter({self.info.column_type})"

    __repr__ = __str__

    @abstractmethod
    def _fit(self) -> "Converter":
        pass

    @abstractmethod
    def _convert(self, flat_arr: flat_arr_type) -> np.ndarray:
        pass

    @abstractmethod
    def _recover(self, flat_arr: flat_arr_type) -> np.ndarray:
        pass

    @property
    def info(self) -> FeatureInfo:
        return self._recognizer.info

    @property
    def converted_input(self) -> np.ndarray:
        if getattr(self, "_converted_features", None) is None:
            self._converted_features = self.convert(self.info.flat_arr)
        return self._converted_features

    def _initialize(self, **kwargs: Any) -> None:
        pass

    def initialize(self) -> None:
        self._initialize(**self._config)
        self._fit()

    def convert(self, flat_arr: flat_arr_type) -> np.ndarray:
        if not self._inplace:
            flat_arr = flat_arr.copy()
        return self._convert(flat_arr)

    def recover(self, flat_arr: flat_arr_type, *, inplace: bool = False) -> np.ndarray:
        if not inplace:
            flat_arr = flat_arr.copy()
        return self._recover(flat_arr)

    recognizer_key = "__recognizer__"
    identifier_key = "__identifier__"

    def dumps_(self) -> Any:
        instance_dict = shallow_copy_dict(self.__dict__)
        instance_dict.pop("_recognizer")
        instance_dict.pop("_converted_features")
        instance_dict[self.identifier_key] = self.__identifier__
        instance_dict[self.recognizer_key] = self._recognizer.dumps()
        return instance_dict

    @classmethod
    def loads(cls, instance_dict: Dict[str, Any], **kwargs: Any) -> "Converter":
        recognizer_data = instance_dict.pop(cls.recognizer_key)
        recognizer = Recognizer.load(data=recognizer_data)
        identifier = instance_dict.pop(cls.identifier_key)
        converter = converter_dict[identifier](recognizer)
        converter.__dict__.update(instance_dict)
        return converter

    @classmethod
    def make_with(
        cls,
        recognizer: Recognizer,
        *,
        inplace: bool = False,
        **kwargs: Any,
    ) -> "Converter":
        key = recognizer.info.column_type.value
        converter = converter_dict[key](recognizer, inplace=inplace, **kwargs)
        converter.initialize()
        return converter

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global converter_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, converter_dict, before_register=before)


__all__ = ["Converter", "converter_dict"]
