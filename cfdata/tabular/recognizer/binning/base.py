import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import register_core

from ...misc import TaskTypes
from ...misc import FeatureInfo


binning_dict: Dict[str, Type["BinningBase"]] = {}


class BinResults(NamedTuple):
    indices: Optional[List[int]]
    values: Union[List[str], List[float]]
    transformed_unique_values: List[int]


class BinningBase:
    def __init__(
        self,
        labels: np.ndarray,
        task_type: TaskTypes,
        config: Dict[str, Any],
    ):
        self.labels = labels
        self.task_type = task_type
        self.config = config

    def binning(
        self,
        info: FeatureInfo,
        sorted_counts: np.ndarray,
        unique_values: Union[List[str], List[float]],
    ) -> BinResults:
        pass

    @classmethod
    def make(
        cls,
        name: str,
        labels: np.ndarray,
        task_type: TaskTypes,
        config: Dict[str, Any],
    ) -> "BinningBase":
        return binning_dict[name](labels, task_type, config)

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global binning_dict
        return register_core(name, binning_dict)


class BinningError(Exception):
    pass


__all__ = ["BinningBase", "BinningError"]
