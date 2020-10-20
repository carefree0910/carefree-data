import numpy as np

from ..base import Processor


@Processor.register("min_max")
class MinMax(Processor):
    @property
    def input_dim(self) -> int:
        return 1

    @property
    def output_dim(self) -> int:
        return 1

    def fit(self, columns: np.ndarray) -> Processor:
        d_min, d_max = columns.min(), columns.max()
        self._caches["min"], self._caches["diff"] = d_min, d_max - d_min
        return self

    def _process(self, columns: np.ndarray) -> np.ndarray:
        d_min, diff = map(self._caches.get, ["min", "diff"])
        columns -= d_min
        columns /= diff
        return columns

    def _recover(self, processed_columns: np.ndarray) -> np.ndarray:
        d_min, diff = map(self._caches.get, ["min", "diff"])
        processed_columns *= diff
        processed_columns += d_min
        return processed_columns


__all__ = ["MinMax"]
