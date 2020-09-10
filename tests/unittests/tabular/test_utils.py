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
        for power in range(3, 6):
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

    def _k_core(self, n_total, iterator):
        te_indices = None
        trx_shape = try_shape = tex_shape = tey_shape = None
        for tr_split, te_split in iterator:
            tr_set, te_set = tr_split.dataset, te_split.dataset
            new_trx_shape, new_try_shape = tr_set.x.shape, tr_set.y.shape
            new_tex_shape, new_tey_shape = te_set.x.shape, te_set.y.shape
            new_te_indices = set(te_split.corresponding_indices)
            self.assertTrue(new_trx_shape[0] + new_tex_shape[0] == n_total)
            self.assertTrue(new_try_shape[0] + new_tey_shape[0] == n_total)
            if te_indices is not None:
                if isinstance(iterator, KFold):
                    self.assertFalse(te_indices & new_te_indices)
                elif isinstance(iterator, KRandom):
                    self.assertTrue(te_indices & new_te_indices)
            te_indices = new_te_indices
            self.assertFalse(set(tr_split.corresponding_indices) & te_indices)
            if trx_shape is None:
                trx_shape, try_shape = new_trx_shape, new_try_shape
                tex_shape, tey_shape = new_tex_shape, new_tey_shape
            else:
                self.assertTrue(trx_shape == new_trx_shape)
                self.assertTrue(try_shape == new_try_shape)
                self.assertTrue(tex_shape == new_tex_shape)
                self.assertTrue(tey_shape == new_tey_shape)

    def test_k_fold(self):
        k = 10
        num_class = 10
        task = TaskTypes.CLASSIFICATION
        for power in range(3, 6):
            n = int(10 ** power)
            x = np.random.random([n, 100]).astype(np_float_type)
            y = np.random.randint(0, num_class, [n, 1])
            k_fold = KFold(k, TabularDataset.from_xy(x, y, task))
            self._k_core(n, k_fold)

    def test_k_random(self):
        k = 10
        test_ratio = 0.1
        num_class = 10
        task = TaskTypes.CLASSIFICATION
        for power in range(3, 6):
            n = int(10 ** power)
            x = np.random.random([n, 100]).astype(np_float_type)
            y = np.random.randint(0, num_class, [n, 1])
            k_random = KRandom(k, test_ratio, TabularDataset.from_xy(x, y, task))
            self._k_core(n, k_random)

    def test_k_bootstrap(self):
        k = 10
        test_ratio = 0.1
        num_class = 10
        task = TaskTypes.CLASSIFICATION
        for power in range(3, 6):
            n = int(10 ** power)
            x = np.random.random([n, 100]).astype(np_float_type)
            y = np.random.randint(0, num_class, [n, 1])
            k_bootstrap = KBootstrap(k, test_ratio, TabularDataset.from_xy(x, y, task))
            self._k_core(n, k_bootstrap)

    def test_imbalance_sampler(self):
        counts = []
        tolerance = 0.01
        for power in range(3, 6):
            n = int(10 ** power)
            x = np.random.random([n, 100]).astype(np_float_type)
            y = (np.random.random([n, 1]) >= 0.95).astype(np_int_type) + 2
            for _ in range((5 - power) * 10 + 1):
                data = TabularData().read(x, y)
                sampler = ImbalancedSampler(data, verbose_level=0)
                counts.append(np.unique(y[sampler.get_indices()], return_counts=True)[1])
        counts = np.vstack(counts)
        ratios = (counts / counts.sum(1, keepdims=True)).T
        diff = np.mean(ratios[1] - ratios[0])
        self.assertLess(diff, tolerance)

    def test_data_loader(self):
        num_class = 10
        n = int(10 ** 5)

        x = np.random.random([n, 100]).astype(np_float_type)
        y = np.random.randint(0, num_class, [n, 1])
        data = TabularData().read(x, y)
        sampler = ImbalancedSampler(data)
        loader = DataLoader(128, sampler)
        x_batch_shape = y_batch_shape = None
        for i, (x_batch, y_batch) in enumerate(loader):
            new_x_shape, new_y_shape = x_batch.shape, y_batch.shape
            if x_batch_shape is None:
                x_batch_shape, y_batch_shape = new_x_shape, new_y_shape
            elif i != len(loader) - 1:
                self.assertTrue(x_batch_shape == new_x_shape)
                self.assertTrue(y_batch_shape == new_y_shape)

    def test_k_fold_sanity(self):
        num_features = 8
        for power in [1, 2, 3]:
            num_samples = int(10 ** power)
            half_samples = num_samples // 2
            num_elem = num_samples * num_features
            x = np.arange(num_elem).reshape([num_samples, num_features])
            y = np.zeros(num_samples, np_int_type)
            # here, number of positive samples will be less than number of negative samples (by 2)
            # hence, one positive sample will be duplicated, and one negative sample will be dropped
            y[-half_samples+1:] = 1
            dataset = TabularDataset.from_xy(x, y, TaskTypes.CLASSIFICATION)
            k_fold = KFold(half_samples, dataset)
            for train_fold, test_fold in k_fold:
                x_stack = np.vstack([train_fold.dataset.x, test_fold.dataset.x])
                x_unique = np.unique(x_stack.ravel())
                self.assertEqual(num_elem - len(x_unique), num_features)
                self.assertTrue(sorted(test_fold.dataset.y.ravel()), [0, 1])
            # here, labels are balanced
            # hence, all folds should cover the entire dataset
            y[-half_samples] = 1
            dataset = TabularDataset.from_xy(x, y, TaskTypes.CLASSIFICATION)
            k_fold = KFold(half_samples, dataset)
            for train_fold, test_fold in k_fold:
                x_stack = np.vstack([train_fold.dataset.x, test_fold.dataset.x])
                x_unique = np.unique(x_stack.ravel())
                self.assertEqual(num_elem, len(x_unique))
                self.assertEqual(sorted(test_fold.dataset.y.ravel()), [0, 1])

    def test_k_random_sanity(self):
        num_features = 8
        for power in [1, 2, 3]:
            num_samples = int(10 ** power)
            num_elem = num_samples * num_features
            x = np.arange(num_elem).reshape([num_samples, num_features])
            y = np.zeros(num_samples, np_int_type)
            # here, we only have one positive sample
            # but we will still have this positive sample in each test fold
            y[-1] = 1
            dataset = TabularDataset.from_xy(x, y, TaskTypes.CLASSIFICATION)
            k_random = KRandom(10, 2, dataset)
            for train_fold, test_fold in k_random:
                x_stack = np.vstack([train_fold.dataset.x, test_fold.dataset.x])
                self.assertEqual(num_elem - len(np.unique(x_stack.ravel())), num_features)
                self.assertTrue(sorted(test_fold.dataset.y.ravel()), [0, 1])


if __name__ == '__main__':
    unittest.main()
