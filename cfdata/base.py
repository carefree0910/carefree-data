from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from cftool.misc import SavingMixin


class DataBase(SavingMixin, metaclass=ABCMeta):
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def read(self, x: Any, y: Any, **kwargs: Any) -> "DataBase":
        pass

    @abstractmethod
    def transform(self, x: Any, y: Any, **kwargs: Any) -> Any:
        pass

    def recover_labels(self, y: Any, *, inplace: bool = False) -> Any:
        pass


__all__ = ["DataBase"]
