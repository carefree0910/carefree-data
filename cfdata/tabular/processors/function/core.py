import numpy as np

from ..base import Processor


@Processor.register("lambda")
class Lambda(Processor):
    def initialize(self):
        self._process = self._config["process"]
        self._recover = self._config["recover"]
        self._input_dim = self._config["input_dim"]
        self._output_dim = self._config["output_dim"]

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def fit(self, columns: np.ndarray) -> Processor:
        return self

    def _process(self, columns: np.ndarray) -> np.ndarray:
        return self._process(columns)

    def _recover(self, processed_columns: np.ndarray) -> np.ndarray:
        return self._recover(processed_columns)


__all__ = ["Lambda"]
