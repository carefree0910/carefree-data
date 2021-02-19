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

from ...misc import FeatureInfo


binning_dict: Dict[str, Type["BinningBase"]] = {}


class BinResults(NamedTuple):
    indices: Optional[List[int]]
    values: Union[List[str], List[float]]
    transformed_unique_values: List[int]


class BinningBase:
    def __init__(self, labels: np.ndarray, config: Dict[str, Any]):
        self.labels = labels
        self.config = config

    def binning(
        self,
        info: FeatureInfo,
        sorted_counts: np.ndarray,
        values: Union[List[str], List[float]],
    ) -> BinResults:
        pass

    @classmethod
    def make(
        cls,
        name: str,
        labels: np.ndarray,
        config: Dict[str, Any],
    ) -> "BinningBase":
        return binning_dict[name](labels, config)

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global binning_dict
        return register_core(name, binning_dict)


__all__ = ["BinningBase"]
