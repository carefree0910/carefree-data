import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union

from .base import BinResults
from .base import BinningBase
from ...misc import TaskTypes
from ...misc import FeatureInfo


@BinningBase.register("fuse")
class FuseBinning(BinningBase):
    def __init__(
        self,
        labels: np.ndarray,
        task_type: TaskTypes,
        config: Dict[str, Any],
    ):
        super().__init__(labels, task_type, config)
        default_bound = config["default_bound"]
        self.num_unique_bound = config["num_unique_bound"]
        self._truncate_ratio = config.setdefault("truncate_ratio", 0.99)
        if self.num_unique_bound is None:
            default_fuse_threshold = 1.0 / default_bound
        else:
            default_fuse_threshold = 1.0 / self.num_unique_bound
        self._fuse_thresh = config.setdefault("fuse_threshold", default_fuse_threshold)
        if self.num_unique_bound is None:
            default_fuse_fix = int(default_bound // 2)
        else:
            default_fuse_fix = int(round(0.5 * self.num_unique_bound))
        self._num_fuse_fix = config.setdefault("num_fuse_fix", default_fuse_fix)

    def binning(
        self,
        info: FeatureInfo,
        sorted_counts: np.ndarray,
        unique_values: Union[List[str], List[float]],
    ) -> BinResults:
        if not info.need_truncate:
            return BinResults(None, unique_values, list(range(len(unique_values))))
        # truncate
        counts_cumsum = np.cumsum(sorted_counts)
        counts_cumsum_ratio = counts_cumsum / counts_cumsum[-1]
        truncate_mask = counts_cumsum_ratio >= self._truncate_ratio
        truncate_idx = np.nonzero(truncate_mask)[0][0]
        unique_values = unique_values[: truncate_idx + 1]
        # fuse
        idx = 0
        cumulate = 0.0
        fused_indices = []
        for i, ratio in enumerate(counts_cumsum_ratio[: truncate_idx + 1]):
            fused_indices.append(idx)
            if i < self._num_fuse_fix or ratio >= self._fuse_thresh + cumulate:
                idx += 1
                cumulate = ratio
        transformed_unique_values = sorted(set(fused_indices))
        return BinResults(fused_indices, unique_values, transformed_unique_values)


__all__ = ["FuseBinning"]
