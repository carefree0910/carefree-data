import numpy as np

from typing import List
from typing import Union

from .base import BinResults
from .base import BinningBase
from ...misc import FeatureInfo


@BinningBase.register("identical")
class IdenticalBinning(BinningBase):
    def binning(
        self,
        info: FeatureInfo,
        sorted_counts: np.ndarray,
        values: Union[List[str], List[float]],
    ) -> BinResults:
        fused_indices = list(range(len(values)))
        transformed_unique_values = fused_indices.copy()
        return BinResults(fused_indices, values, transformed_unique_values)


__all__ = ["IdenticalBinning"]
