import unittest

from cfdata.misc.c import is_all_numeric
from cfdata.misc.c.cython_wrappers import *
from cfdata.misc.c.cython_substitute import *
from cfdata.misc.toolkit import timeit

types = ["used", "c", "naive"]


class TestC(unittest.TestCase):
    def test_is_all_numeric(self):
        multiple = 10 ** 5
        true_arr = ["1", 2, 3.4, "5.6"] * multiple
        false_arr = ["1", 2, 3.4, "5.6", "7.8.9"] * multiple
        methods = [is_all_numeric, c_is_all_numeric, naive_is_all_numeric]
        for t, m in zip(types, methods):
            with timeit(t):
                self.assertTrue(m(true_arr))
                self.assertFalse(m(false_arr))


if __name__ == '__main__':
    unittest.main()
