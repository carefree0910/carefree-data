import numpy as np

from ..base import Processor


@Processor.register("identical")
class Identical(Processor):
    @property
    def input_dim(self) -> int:
        return 1

    @property
    def output_dim(self) -> int:
        return 1

    def fit(self, columns: np.ndarray) -> Processor:
        return self

    def _process(self, columns: np.ndarray) -> np.ndarray:
        return columns

    def _recover(self, processed_columns: np.ndarray) -> np.ndarray:
        return processed_columns


__all__ = ["Identical"]
