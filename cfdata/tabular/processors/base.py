import os

import numpy as np

from typing import *
from abc import ABC, abstractmethod
from cftool.misc import register_core
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.misc import SavingMixin

processor_dict: Dict[str, Type["Processor"]] = {}


class Processor(SavingMixin, ABC):
    def __init__(self,
                 previous_processors: List["Processor"],
                 *,
                 inplace: bool = False,
                 **kwargs):
        self._caches = {}
        self._config = kwargs
        self._inplace = inplace
        self._previous_processors = previous_processors
        start_idx = sum([processor.input_dim for processor in self._previous_processors])
        self._col_indices = [start_idx + i for i in range(self.input_dim)]

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

    @property
    def cache_excludes(self):
        return {"_previous_processors"}

    @property
    def data_tuple_base(self) -> Optional[Type[NamedTuple]]:
        return

    @property
    def data_tuple_attributes(self) -> Optional[List[str]]:
        return

    def initialize(self) -> None:
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

    identifier_file = "identifier.txt"

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
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)

    @classmethod
    def load(cls,
             folder: str,
             *,
             previous_processors: List["Processor"] = None,
             compress: bool = True):
        if previous_processors is None:
            raise ValueError("`previous_processors` must be provided")
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
                processor = processor_dict[identifier]([])
                Saving.load_instance(processor, folder, log_method=processor.log_msg)
                processor._previous_processors = previous_processors
        return processor

    @classmethod
    def make_with(cls,
                  previous_processors: List["Processor"],
                  *,
                  inplace: bool = False,
                  **kwargs) -> "Processor":
        instance = cls(previous_processors, inplace=inplace, **kwargs)
        instance.initialize()
        return instance

    @classmethod
    def register(cls, name):
        global processor_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, processor_dict, before_register=before)


__all__ = ["Processor", "processor_dict"]
