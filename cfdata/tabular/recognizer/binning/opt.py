import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from optbinning import OptimalBinning

from .base import BinResults
from .base import BinningBase
from ...misc import is_int
from ...misc import FeatureInfo


@BinningBase.register("opt")
class OptBinning(BinningBase):
    def __init__(self, labels: np.ndarray, config: Dict[str, Any]):
        super().__init__(labels, config)
        opt_config = config.setdefault("opt_config", {})
        opt_config["dtype"] = "categorical"
        opt_config.setdefault("solver", "mip")
        opt_config.setdefault("cat_cutoff", 0.1)
        self.opt = OptimalBinning(**opt_config)

    def binning(
        self,
        info: FeatureInfo,
        sorted_counts: np.ndarray,
        values: Union[List[str], List[float]],
    ) -> BinResults:
        self.opt.fit(info.flat_arr, self.labels.ravel())
        new_values = []
        fused_indices = []
        transformed_unique_values = []
        for i, split in enumerate(self.opt.splits):
            fused_indices.extend([i] * len(split))
            if is_int(split.dtype):
                split = split.astype(np.float32)
            new_values.extend(split.tolist())
            transformed_unique_values.append(i)
        return BinResults(fused_indices, new_values, transformed_unique_values)


__all__ = ["OptBinning"]
