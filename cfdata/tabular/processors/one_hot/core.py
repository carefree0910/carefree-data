import numpy as np

from sklearn.preprocessing import OneHotEncoder

from ..base import Processor


@Processor.register("one_hot")
class OneHot(Processor):
    def _initialize(self, **kwargs) -> None:
        self._categories = kwargs.get("categories", "auto")

    @property
    def input_dim(self) -> int:
        return 1

    @property
    def output_dim(self) -> int:
        return len(self._all_categories)

    def fit(self,
            columns: np.ndarray) -> Processor:
        self._encoder = OneHotEncoder(categories=self._categories, sparse=False, dtype=np.float32)
        self._encoder.fit(columns)
        self._all_categories = self._encoder.categories_[0]
        return self

    def _process(self,
                 columns: np.ndarray) -> np.ndarray:
        return self._encoder.transform(columns)

    def _recover(self,
                 processed_columns: np.ndarray) -> np.ndarray:
        return self._encoder.inverse_transform(processed_columns)


__all__ = ["OneHot"]
