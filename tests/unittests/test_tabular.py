import os
import math
import unittest

import numpy as np

from cftool.misc import shallow_copy_dict

from cfdata.tabular import *
from cfdata.tabular.types import *


class TestTabularData(unittest.TestCase):
    x = y = y_bundle = task_types = cannot_regressions = str_columns = cat_columns = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.x = [
            [1, "1.0", "nan", math.nan, 0, math.nan, 1, "nan", "a", "0", 0.2, "aa", "1", "1", "aa"],
            [-1, "-1.0", "aaa", 1, 0.1, math.nan, 0, "nan", "b", "0.1", math.nan, "bb", "2", "1", "aa"],
            [1, "-1.0", "nan", 1, 0, math.nan, 0., "nan", "a", "nan", 0, "cc", "3", "1", "aa"],
            [-1, "1.0", "bbb", 1, 10, math.nan, 1., "nan", "c", "0.2", 0.1, "dd", "4", "1", "aa"],
            [1, "-1.0", "nan", 1, 1, math.nan, 1., "nan", "a", "0.5", -0.1, "ee", "5", "1", "aa"],
            [-1, "1.0", "aaa", math.nan, 0, math.nan, -1., "nan", "b", "nan", math.nan, "ff", "6", "1", "aa"],
            [-1, "1.0", "aaa", math.nan, 0, math.nan, -1., "nan", "b", "nan", math.nan, "gg", "7", "1", "aa"]
        ]
        cls.str_columns = [2, 8, 11, 14]
        cls.cat_columns = [0, 1, 3, 5, 6, 7, 12, 13]
        cls.y_bundle = [
            np.atleast_2d([1, 2, 3, 4, 3, 4, 4]).T,
            np.atleast_2d([1, 2, 3, 4, 3, 4, 4.1]).T,
            np.atleast_2d([1, 2.3, 3.4, 4.5, 5.6]).T,
            np.atleast_2d(["1", "2", "3", "4", "3", "4"]).T.tolist(),
            np.atleast_2d(["1", "2", "3", "4", "3", "4", "4"]).T,
            np.atleast_2d(["1", "2.3", "3.4", "4.5", "5.6", "6.7"]).T.tolist(),
            np.atleast_2d(["1", "2.3", "3.4", "4.5", "5.6", "6.7.8"]).T.tolist(),
            np.atleast_2d(["1.0", "2.0", "3.0", "4.0", "3.0", "4.0"]).T,
            np.atleast_2d(["1.0", "2.0", "3.0", "4.0", "3.0", "4.1"]).T,
            np.atleast_2d(["one", "two", "one", "two", "two", "one"]).T,
            np.atleast_2d(["one", "two", "two", "one", "two", "one", "one"]).T.tolist()
        ]
        cls.task_types = [
            TaskTypes.CLASSIFICATION,
            TaskTypes.REGRESSION,
            TaskTypes.REGRESSION,
            TaskTypes.CLASSIFICATION,
            TaskTypes.CLASSIFICATION,
            TaskTypes.REGRESSION,
            TaskTypes.CLASSIFICATION,
            TaskTypes.CLASSIFICATION,
            TaskTypes.REGRESSION,
            TaskTypes.CLASSIFICATION,
            TaskTypes.CLASSIFICATION
        ]
        cls.cannot_regressions = {6, 9, 10}

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.x, cls.y_bundle, cls.task_types, cls.str_columns, cls.cat_columns, cls.cannot_regressions

    @property
    def y_np(self):
        return np.array(self.y)

    def _get_data(self, **kwargs):
        data = TabularData(**kwargs).read(self.x, self.y)
        return data

    def _same_with_y_np(self, data):
        new_y = data.recover_labels(data.processed.y)
        new = DataTuple([[0]], new_y)
        y_np = self.y_np
        if data.recognizers[-1].info.column_type is not ColumnTypes.STRING:
            y_np = y_np.astype(np.float32)
        original = DataTuple([[0]], y_np)
        return new == original

    def _test_core(self, data_config):
        for i, y in enumerate(self.y_bundle):
            self.y = y
            preset_task_type = self.task_types[i]
            for task_type in list(TaskTypes) + [None]:
                local_config = shallow_copy_dict(data_config)
                local_config["task_type"] = task_type
                if task_type is TaskTypes.REGRESSION and i in self.cannot_regressions:
                    with self.assertRaises(ValueError):
                        self._get_data(**local_config)
                else:
                    data = self._get_data(**local_config)
                    if task_type is not None:
                        self.assertTrue(data.task_type is task_type)
                    else:
                        self.assertTrue(data.task_type is preset_task_type)
                    self.assertTrue(data.transform(self.x, self.y) == data.processed)
                    self.assertTrue(self._same_with_y_np(data))

    def test_read_features_only(self):
        self.y = None
        data = self._get_data()
        self.assertTrue(data.transform(self.x, self.y) == data.processed)

    def test_read_from_list_with_column_info(self):
        self._test_core({
            "string_columns": self.str_columns,
            "categorical_columns": self.cat_columns
        })

    def test_read_from_list_without_column_info(self):
        self._test_core({})

    def test_save_and_load(self):
        task = "mnist_small"
        task_file = os.path.join("data", f"{task}.txt")
        data = TabularData().read(task_file).save(task)
        loaded = TabularData().load(task)
        self.assertTrue(data == loaded)
        self.assertTrue(loaded.transform(task_file) == data.processed)
        os.remove(f"{task}.zip")


if __name__ == '__main__':
    unittest.main()
