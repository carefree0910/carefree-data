from typing import Any
from cftool.misc import SavingMixin
from abc import abstractmethod, ABCMeta


class DataBase(SavingMixin, metaclass=ABCMeta):
    @abstractmethod
    def __eq__(self, other: "DataBase"):
        pass

    @abstractmethod
    def read(self, x: Any, y: Any, **kwargs) -> "DataBase":
        pass

    @abstractmethod
    def transform(self, x: Any, y: Any, **kwargs) -> Any:
        pass

    def recover_labels(self, y: Any, *, inplace: bool = False) -> Any:
        pass


__all__ = ["DataBase"]
