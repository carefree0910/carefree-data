import numpy as np

from ..base import Processor


@Processor.register("normalize")
class Normalize(Processor):
    @property
    def input_dim(self) -> int:
        return 1

    @property
    def output_dim(self) -> int:
        return 1

    def initialize(self) -> None:
        self._eps = self._config.get("eps", 1e-8)

    def fit(self, columns: np.ndarray) -> Processor:
        self._caches["mean"] = columns.mean()
        self._caches["std"] = columns.std() + self._eps
        return self

    def _process(self, columns: np.ndarray) -> np.ndarray:
        mean, std = map(self._caches.get, ["mean", "std"])
        columns -= mean
        columns /= std
        return columns

    def _recover(self, processed_columns: np.ndarray) -> np.ndarray:
        mean, std = map(self._caches.get, ["mean", "std"])
        processed_columns *= std
        processed_columns += mean
        return processed_columns


__all__ = ["Normalize"]
