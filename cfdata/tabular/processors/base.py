import numpy as np

from typing import *
from abc import ABC, abstractmethod

from ...misc.toolkit import register_core

processor_dict: Dict[str, Type["Processor"]] = {}


class Processor(ABC):
    def __init__(self,
                 previous_processors: List["Processor"],
                 *,
                 inplace: bool = False,
                 **kwargs):
        self._caches = {}
        self._inplace = inplace
        self._previous_processors = previous_processors
        start_idx = sum([processor.input_dim for processor in self._previous_processors])
        self._col_indices = [start_idx + i for i in range(self.input_dim)]
        self._initialize(**kwargs)

    def __str__(self):
        return f"{type(self).__name__}()"

    __repr__ = __str__

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def fit(self,
            columns: np.ndarray) -> "Processor":
        pass

    @abstractmethod
    def _process(self,
                 columns: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _recover(self,
                 processed_columns: np.ndarray) -> np.ndarray:
        pass

    @property
    def input_indices(self) -> List[int]:
        return self._col_indices

    @property
    def output_indices(self) -> List[int]:
        previous_dimensions = sum([method.output_dim for method in self._previous_processors])
        return list(range(previous_dimensions, previous_dimensions + self.output_dim))

    def _initialize(self, **kwargs) -> None:
        pass

    def process(self,
                columns: np.ndarray) -> np.ndarray:
        if not self._inplace:
            columns = columns.copy()
        return self._process(columns)

    def recover(self,
                columns: np.ndarray,
                *,
                inplace: bool = False) -> np.ndarray:
        if not inplace:
            columns = columns.copy()
        return self._recover(columns)

    @classmethod
    def register(cls, name):
        global processor_dict
        return register_core(name, processor_dict)


__all__ = ["Processor", "processor_dict"]
