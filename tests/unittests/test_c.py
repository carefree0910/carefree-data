import unittest
import numpy as np

from cftool.misc import timeit, allclose

from cfdata.types import *
from cfdata.misc.c import *
from cfdata.misc.c.cython_wrappers import *
from cfdata.misc.c.cython_substitute import *

types = ["used", "c", "naive"]


class TestC(unittest.TestCase):
    arr1 = arr2 = arr3 = None

    @classmethod
    def setUpClass(cls) -> None:
        multiple = 10 ** 5
        cls.arr1 = ["1", 2, 3.4, "5.6"] * multiple
        cls.arr2 = ["1", 2, 3.4, "5.6", "7.8.9"] * multiple
        cls.arr3 = [1, 2, 3.4, 5.6, 7.8] * multiple

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.arr1, cls.arr2, cls.arr3

    @staticmethod
    def _print_header(title):
        print("\n".join(["=" * 100, title, "-" * 100]))

    def test_is_all_numeric(self):
        self._print_header("is_all_numeric")
        methods = [is_all_numeric, c_is_all_numeric, naive_is_all_numeric]
        for t, m in zip(types, methods):
            with timeit(t):
                self.assertTrue(m(self.arr1))
                self.assertFalse(m(self.arr2))

    def test_flat_arr_to_float32(self):
        self._print_header("flat_arr_to_float32")
        results = []
        methods = [
            flat_arr_to_float32,
            c_flat_arr_to_float32,
            naive_flat_arr_to_float32,
        ]
        for t, m in zip(types, methods):
            with timeit(t):
                results.append(m(self.arr1))
        self.assertTrue(allclose(*results))

    def test_transform_flat_data_with_dict(self):
        self._print_header("transform_flat_data_with_dict")
        results = []
        arr = np.array(self.arr3, dtype=np_float_type)
        transform_dict = {1: 0, 2: 1, 3.4: 2, 5.6: 3, 7.8: 4}
        methods = [
            transform_flat_data_with_dict,
            c_transform_flat_data_with_dict,
            naive_transform_flat_data_with_dict,
        ]
        for t, m in zip(types, methods):
            with timeit(t):
                results.append(m(arr, transform_dict, False))
        self.assertTrue(allclose(*results))


if __name__ == "__main__":
    unittest.main()
