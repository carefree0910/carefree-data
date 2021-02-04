import os
import math
import unittest

import numpy as np

from cftool.misc import shallow_copy_dict

from cfdata.types import *
from cfdata.tabular import *
from cfdata.tabular.misc import *

file_folder = os.path.dirname(__file__)
data_folder = os.path.abspath(os.path.join(file_folder, os.pardir, "data"))


class TestTabularData(unittest.TestCase):
    x_ts = y_ts = ts_config = None
    x = y = y_bundle = task_types = None
    cannot_regressions = str_columns = cat_columns = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.x_ts = [
            ["orange", "2020-01-01", 1.5],
            ["apple", "2020-01-01", 2.5],
            ["banana", "2020-01-01", 2.0],
            ["pear", "2020-01-01", 1.0],
            ["orange", "2020-01-03", 2.5],
            ["apple", "2020-01-03", 1.5],
            ["banana", "2020-01-03", 1.0],
            ["pear", "2020-01-03", 2.0],
            ["orange", "2020-01-02", 2.0],
            ["apple", "2020-01-02", 2.0],
            ["banana", "2020-01-02", 1.5],
        ]
        cls.y_ts = np.atleast_2d(
            [2.0, 2.0, 2.5, 1.5, 2.0, 2.0, 1.5, 2.5, 2.5, 2.5, 2.0, 2.0]
        ).T
        cls.ts_config = TimeSeriesConfig(id_column_idx=0, time_column_idx=1)
        cls.x = [
            [
                1,
                "1.0",
                "nan",
                math.nan,
                0,
                math.nan,
                1,
                "nan",
                "a",
                "0",
                0.2,
                "aa",
                "1",
                "1",
                "aa",
            ],
            [
                -1,
                "-1.0",
                "aaa",
                1,
                0.1,
                math.nan,
                0,
                "nan",
                "b",
                "0.1",
                math.nan,
                "bb",
                "2",
                "1",
                "aa",
            ],
            [
                1,
                "-1.0",
                "nan",
                1,
                0,
                math.nan,
                0.0,
                "nan",
                "a",
                "nan",
                0,
                "cc",
                "3",
                "1",
                "aa",
            ],
            [
                -1,
                "1.0",
                "bbb",
                1,
                10,
                math.nan,
                1.0,
                "nan",
                "c",
                "0.2",
                0.1,
                "dd",
                "4",
                "1",
                "aa",
            ],
            [
                1,
                "-1.0",
                "nan",
                1,
                1,
                math.nan,
                1.0,
                "nan",
                "a",
                "0.5",
                -0.1,
                "ee",
                "5",
                "1",
                "aa",
            ],
            [
                -1,
                "1.0",
                "aaa",
                math.nan,
                0,
                math.nan,
                -1.0,
                "nan",
                "b",
                "nan",
                math.nan,
                "ff",
                "6",
                "1",
                "aa",
            ],
            [
                -1,
                "1.0",
                "aaa",
                math.nan,
                0,
                math.nan,
                -1.0,
                "nan",
                "b",
                "nan",
                math.nan,
                "gg",
                "7",
                "1",
                "aa",
            ],
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
            np.atleast_2d(["one", "two", "two", "one", "two", "one", "one"]).T.tolist(),
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
            TaskTypes.CLASSIFICATION,
        ]
        cls.cannot_classifications = {1, 2}

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.x_ts, cls.y_ts, cls.ts_config
        del (
            cls.x,
            cls.y_bundle,
            cls.task_types,
            cls.str_columns,
            cls.cat_columns,
            cls.cannot_regressions,
        )

    def _get_data(self, y, **kwargs):
        data = TabularData(**kwargs).read(self.x, y)
        return data

    @staticmethod
    def _same_with_y(data, y):
        new_y = data.recover_labels(data.processed.y)
        new = DataTuple([[0]], new_y)
        y_np = np.array(y)
        if data.recognizers[-1].info.column_type is not ColumnTypes.STRING:
            y_np = y_np.astype(np_float_type)
        original = DataTuple([[0]], y_np)
        return new == original

    def _test_core(self, data_config):
        for i, y in enumerate(self.y_bundle):
            preset_task_type = self.task_types[i]
            for task_type in [TaskTypes.CLASSIFICATION, TaskTypes.REGRESSION]:
                local_config = shallow_copy_dict(data_config)
                local_config["task_type"] = task_type
                cannot_clf = i in self.cannot_classifications
                if task_type is TaskTypes.CLASSIFICATION and cannot_clf:
                    with self.assertRaises(Exception):
                        data = self._get_data(y, **local_config)
                        self.assertTrue(self._same_with_y(data, y))
                else:
                    data = self._get_data(y, **local_config)
                    if not task_type.is_none:
                        self.assertTrue(data.task_type is task_type)
                    else:
                        self.assertTrue(data.task_type is preset_task_type)
                    self.assertTrue(data.transform(self.x, y) == data.processed)
                    self.assertTrue(self._same_with_y(data, y))

    def test_read_features_only(self):
        data = self._get_data(None)
        self.assertTrue(data.transform(self.x, None) == data.processed)

    def test_read_from_list_with_column_info(self):
        self._test_core(
            {
                "string_columns": self.str_columns,
                "categorical_columns": self.cat_columns,
            }
        )

    def test_read_from_list_without_column_info(self):
        self._test_core({})

    def test_save_and_load(self):
        task = "mnist_small"
        task_file = os.path.join(data_folder, f"{task}.txt")
        data = TabularData().read(task_file).save(task)
        copied = data.copy_to(task_file)
        self.assertTrue(data == copied)
        loaded = TabularData.load(task)
        self.assertTrue(data == loaded)
        self.assertTrue(loaded.transform(task_file) == data.processed)
        simplified_file = f"{task}_simplified"
        data.save(simplified_file, retain_data=False)
        loaded_simplified = TabularData.load(simplified_file)
        self.assertTrue(loaded_simplified.transform(task_file) == data.processed)
        os.remove(f"{task}.zip")
        os.remove(f"{simplified_file}.zip")

    def _test_recover_labels_core(self, dataset):
        data = TabularData.from_dataset(dataset)
        dataset_processed = data.to_dataset()
        recovered = data.recover_labels(dataset_processed.y)
        self.assertTrue(np.allclose(recovered, dataset.y))

    def test_recover_labels(self):
        self._test_recover_labels_core(TabularDataset.iris())
        self._test_recover_labels_core(TabularDataset.boston())
        self._test_recover_labels_core(TabularDataset.digits())
        self._test_recover_labels_core(TabularDataset.breast_cancer())

    def _test_recover_features_core(self, dataset):
        column_indices = list(range(dataset.num_features))
        data = TabularData.from_dataset(dataset)
        dataset_processed = data.to_dataset()
        dataset_xt = dataset.x.T
        dataset_processed_x = dataset_processed.x
        for col_idx in column_indices:
            if col_idx in data.excludes:
                continue
            processor = data.processors[col_idx]
            columns = dataset_processed_x[..., processor.output_indices]
            processor_recovered = processor.recover(columns)
            converter = data.converters[col_idx]
            recovered = converter.recover(processor_recovered.ravel())
            original = dataset_xt[col_idx]
            try:
                self.assertTrue(np.allclose(recovered, original, atol=1e-5))
            except AssertionError:
                different = np.nonzero(recovered != original)[0]
                supported = converter._transform_dict
                reversed_transform = {}
                for k, v in supported.items():
                    reversed_transform.setdefault(v, set()).add(k)
                oob_value = converter._reverse_transform_dict[0.0]
                for idx in different:
                    original_item = original[idx].item()
                    recovered_item = recovered[idx].item()
                    transformed_idx = supported.get(original_item)
                    if transformed_idx is None:
                        self.assertTrue(recovered_item == oob_value)
                    else:
                        reverse_supported = reversed_transform[transformed_idx]
                        self.assertTrue(recovered_item in reverse_supported)

    def test_recover_features(self):
        self._test_recover_features_core(TabularDataset.iris())
        self._test_recover_features_core(TabularDataset.boston())
        self._test_recover_features_core(TabularDataset.digits())
        self._test_recover_features_core(TabularDataset.breast_cancer())

    def _test_equal_core(self, dataset):
        d1 = TabularData().read(*dataset.xy)
        d2 = TabularData.from_dataset(dataset)
        self.assertTrue(d1 == d2)

    def test_equal(self):
        self._test_equal_core(TabularDataset.iris())
        self._test_equal_core(TabularDataset.boston())
        self._test_equal_core(TabularDataset.digits())
        self._test_equal_core(TabularDataset.breast_cancer())

    def test_from_str(self):
        self.assertTrue(TaskTypes.from_str("") is TaskTypes.NONE)
        self.assertTrue(TaskTypes.from_str("reg") is TaskTypes.REGRESSION)
        self.assertTrue(TaskTypes.from_str("clf") is TaskTypes.CLASSIFICATION)
        self.assertTrue(TaskTypes.from_str("ts_clf") is TaskTypes.TIME_SERIES_CLF)
        self.assertTrue(TaskTypes.from_str("ts_reg") is TaskTypes.TIME_SERIES_REG)

    def test_split_data_tuple_with_indices(self):
        lst, npy = list(range(10)), np.arange(10)
        dt = DataTuple(lst, None)
        self.assertEqual(dt.split_with([2, 4, 6]).x, [2, 4, 6])
        dt = DataTuple(npy, None)
        self.assertTrue(np.allclose(dt.split_with([2, 4, 6]).x, [2, 4, 6]))
        dt = DataTuple(None, lst)
        self.assertEqual(dt.split_with([1, 3, 5]).y, [1, 3, 5])
        dt = DataTuple(None, npy)
        self.assertTrue(np.allclose(dt.split_with([1, 3, 5]).y, [1, 3, 5]))
        dt = DataTuple.with_transpose(list(map(list, zip(lst))), None)
        self.assertEqual(dt.split_with([7, 8, 9]).xT, [[7, 8, 9]])
        dt = DataTuple.with_transpose(npy[..., None], None)
        self.assertTrue(np.allclose(dt.split_with([7, 8, 9]).xT, [[7, 8, 9]]))

    def test_quote(self):
        data_file = os.path.join(data_folder, "quote_test.csv")
        data = TabularData().read(data_file)
        gt = {0: "f1", 1: "f2", 2: "f3", 3: "f4", 4: "f5"}
        self.assertDictEqual(data.column_names, gt)
        self.assertListEqual(data.raw.x[0].tolist(), [3, "2, 3", '4"', 5])
        self.assertListEqual(data.raw.y[0].tolist(), [0])

    def test_ts_split(self):
        data = TabularData(time_series_config=self.ts_config).read(self.x_ts, self.y_ts)
        split = data.split(5).split.raw.xT[1]
        self.assertListEqual(split, ["2020-01-02"] + ["2020-01-03"] * 4)
        for _ in range(100):
            sampler = ImbalancedSampler(data, aggregation_config={"num_history": 2})
            loader = DataLoader(2, sampler, return_indices=True)
            for _, indices_batch in loader:
                for indices in indices_batch:
                    self.assertEqual(self.x_ts[indices[0]][0], self.x_ts[indices[1]][0])
            sampler = ImbalancedSampler(data, aggregation_config={"num_history": 3})
            loader = DataLoader(2, sampler, return_indices=True)
            for _, indices_batch in loader:
                for indices in indices_batch:
                    self.assertNotEqual(self.x_ts[indices[0]][0], "pear")
                    self.assertNotEqual(self.x_ts[indices[1]][0], "pear")
                    self.assertEqual(self.x_ts[indices[0]][0], self.x_ts[indices[1]][0])

    def test_ts_sorting_indices(self):
        shuffled_indices = np.random.permutation(len(self.x_ts))
        x_ts = [self.x_ts[i] for i in shuffled_indices]
        y_ts = [self.y_ts[i].tolist() for i in shuffled_indices]
        data = TabularData(time_series_config=self.ts_config).read(x_ts, y_ts)
        gt = [
            "2020-01-01",
            "2020-01-01",
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
            "2020-01-02",
            "2020-01-03",
            "2020-01-03",
            "2020-01-03",
            "2020-01-03",
        ]
        self.assertListEqual([x_ts[i][1] for i in data.ts_sorting_indices], gt)

    def test_simplify(self):
        n = 1000000
        x = np.random.random([n, 5])
        y = np.random.randint(0, 2, [n, 1])
        export_name = "test_data"
        simplified_export_name = f"{export_name}_simple"
        d = TabularData().read(x, y)
        d1 = TabularData.simple("clf", simplify=True)
        d1.read(x, y)
        d.save(export_name)
        d2 = TabularData.load(export_name)
        self.assertTrue(d == d2)
        d.save(export_name, retain_data=False)
        d1.save(simplified_export_name, retain_data=False)
        d2 = TabularData.load(export_name)
        d3 = TabularData.load(simplified_export_name)
        self.assertTrue(np.allclose(x, d1.transform(x).x))
        self.assertFalse(np.allclose(x, d.transform(x).x))
        self.assertTrue(np.allclose(y, d1.transform_labels(y)))
        self.assertTrue(d.transform(x) == d2.transform(x))
        self.assertTrue(d1.transform(x) == d3.transform(x))
        os.remove(f"{export_name}.zip")
        os.remove(f"{simplified_export_name}.zip")


if __name__ == "__main__":
    unittest.main()
