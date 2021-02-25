import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from optbinning import OptimalBinning
from optbinning import ContinuousOptimalBinning
from optbinning import MulticlassOptimalBinning
from cftool.misc import shallow_copy_dict

from .base import BinResults
from .base import BinningBase
from ...misc import is_float
from ...misc import TaskTypes
from ...misc import FeatureInfo


@BinningBase.register("opt")
class OptBinning(BinningBase):
    def __init__(
        self,
        labels: np.ndarray,
        task_type: TaskTypes,
        config: Dict[str, Any],
    ):
        super().__init__(labels, task_type, config)
        self.opt_config = config.setdefault("opt_config", {})

    def binning(
        self,
        info: FeatureInfo,
        sorted_counts: np.ndarray,
        unique_values: Union[List[str], List[float]],
    ) -> BinResults:
        x = info.flat_arr
        y = self.labels.ravel()
        opt_config = shallow_copy_dict(self.opt_config)
        assert isinstance(x, np.ndarray)
        # feature type
        if is_float(x.dtype):  # type: ignore
            opt_config["dtype"] = "numerical"
            opt_config.setdefault("solver", "cp")
        else:
            opt_config["dtype"] = "categorical"
            opt_config.setdefault("solver", "mip")
            opt_config.setdefault("cat_cutoff", 0.1)
        # task type
        if self.task_type.is_reg:
            opt_config.pop("solver")
            base = ContinuousOptimalBinning
        else:
            if int(round(y.max().item())) == 1:
                base = OptimalBinning
            else:
                opt_config.pop("dtype")
                opt_config.pop("cat_cutoff", None)
                base = MulticlassOptimalBinning
                td = {v: i for i, v in enumerate(unique_values)}
                unique_values = [float(td[v]) for v in unique_values]
                x = np.array([td[v] for v in x], np.float32)
        # core
        opt = base(**opt_config).fit(x, y)
        fused_indices = opt.transform(unique_values, metric="indices")
        transformed_unique_values = sorted(set(fused_indices))
        if len(transformed_unique_values) <= 1:
            msg = "not more than 1 unique values are left after `OptBinning` is applied"
            raise ValueError(msg)
        return BinResults(fused_indices, unique_values, transformed_unique_values)


__all__ = ["OptBinning"]
