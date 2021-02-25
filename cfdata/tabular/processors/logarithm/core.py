import numpy as np

from ..base import Processor


@Processor.register("logarithm")
class Logarithm(Processor):
    @property
    def input_dim(self) -> int:
        return 1

    @property
    def output_dim(self) -> int:
        return 1

    def fit(self, columns: np.ndarray) -> Processor:
        return self

    def _process(self, columns: np.ndarray) -> np.ndarray:
        sign = np.sign(columns)
        return np.log(columns * sign + 1.0) * sign

    def _recover(self, processed_columns: np.ndarray) -> np.ndarray:
        sign = np.sign(processed_columns)
        return (np.exp(processed_columns * sign) - 1.0) * sign


__all__ = ["Logarithm"]
