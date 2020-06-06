import operator
import unittest

import numpy as np

from cfdata.types import *
from cfdata.tabular import *
from cfdata.tabular.types import *


class TestTabularUtils(unittest.TestCase):
    def test_data_splitter(self):
        def _get_proportion(*y_):
            ratio_arr = []
            for yy_ in y_:
                yy_ = yy_.ravel()
                ratio_arr.append(np.bincount(yy_) / len(yy_))
            return np.array(ratio_arr, np.float32)
        def _mae(arr1, arr2):
            return np.abs(arr1 - arr2).mean().item()
        def _assert_proportion(n_, p1, p2):
            mae_ = _mae(p1, p2)
            self.assertTrue(n_ * mae_ <= tolerance)

        n_class = 10
        tolerance = 5
        ratios = [0.1, 0.1]
        task = TaskTypes.CLASSIFICATION
        for power in range(3, 7):
            n = int(10 ** power)
            n_cv, n_test = map(int, map(operator.mul, 2 * [n], ratios))
            x = np.random.random([n, 100]).astype(np_float_type)
            y = np.random.randint(0, n_class, [n, 1])
            data_splitter = DataSplitter().fit(TabularDataset.from_xy(x, y, task))
            original_proportion = _get_proportion(y)
            # we should split test set before cv set
            x_test, y_test = data_splitter.split(n_test).dataset.xy
            x_cv, y_cv = data_splitter.split(n_cv).dataset.xy
            x_train, y_train = data_splitter.remained_xy
            new_proportion1 = _get_proportion(y_train, y_cv, y_test)
            # we can simplify our codes by calling `split_multiple`
            data_splitter.reset()
            results = data_splitter.split_multiple([n_test, n_cv], return_remained=True)
            y_test, y_cv, y_train = [result.dataset.y for result in results]
            new_proportion2 = _get_proportion(y_train, y_cv, y_test)
            # ratio is supported too
            data_splitter.reset()
            results = data_splitter.split_multiple(ratios, return_remained=True)
            y_test, y_cv, y_train = [result.dataset.y for result in results]
            new_proportion3 = _get_proportion(y_train, y_cv, y_test)
            _assert_proportion(n, original_proportion, new_proportion1)
            _assert_proportion(n, original_proportion, new_proportion2)
            _assert_proportion(n, original_proportion, new_proportion3)
            _assert_proportion(n, new_proportion1, new_proportion2)
            _assert_proportion(n, new_proportion2, new_proportion3)
            _assert_proportion(n, new_proportion3, new_proportion1)


if __name__ == '__main__':
    unittest.main()
