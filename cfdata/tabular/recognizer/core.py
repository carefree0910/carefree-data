import math
import numpy as np

from typing import *
from collections import Counter

from ..types import *
from ...misc.c import *
from ...misc.toolkit import get_counter_from_arr


class Recognizer:
    def __init__(self,
                 column_name: str,
                 *,
                 is_label: bool = False,
                 task_type: TaskTypes = None,
                 force_valid: bool = False,
                 force_string: bool = False,
                 force_numerical: bool = False,
                 force_categorical: bool = False,
                 numerical_threshold: float = 0.5):
        self.name = column_name
        self.is_label = is_label
        self.task_type = task_type
        self.force_valid = force_valid
        self.force_string = force_string
        self.force_numerical = force_numerical
        self.force_categorical = force_categorical
        self.numerical_threshold = numerical_threshold
        self._info = self._counter = self._transform_dict = None

    def __str__(self):
        return f"Recognizer({self.info.column_type})"

    __repr__ = __str__

    @property
    def info(self) -> FeatureInfo:
        return self._info

    @property
    def counter(self) -> Counter:
        return self._counter

    @property
    def transform_dict(self) -> Dict[Union[int, str], int]:
        return self._transform_dict

    def _make_invalid_info(self,
                           msg: str,
                           contains_nan: bool,
                           nan_mask: np.ndarray) -> "Recognizer":
        self._info = FeatureInfo(contains_nan, None, nan_mask=nan_mask, is_valid=False, msg=msg)
        return self

    @staticmethod
    def _make_string_info(flat_arr, is_valid, msg, unique_values=None) -> FeatureInfo:
        return FeatureInfo(
            False, flat_arr, is_valid=is_valid, msg=msg, need_transform=True,
            column_type=ColumnTypes.STRING, unique_values_sorted_by_counts=unique_values
        )

    def _check_string_column(self,
                             flat_arr: flat_arr_type) -> Tuple[bool, Union[FeatureInfo, None]]:
        if self.force_numerical or self.force_categorical:
            return False, None
        all_numeric = is_all_numeric(flat_arr)
        is_regression = self.task_type is TaskTypes.REGRESSION
        if self.is_label and is_regression and not all_numeric:
            raise ValueError("task_type is REGRESSION but labels are not all numeric")
        if all_numeric:
            return False, None
        self._counter = get_counter_from_arr(flat_arr)
        unique_values = [elem[0] for elem in self._counter.most_common()]
        self._transform_dict = {v: i for i, v in enumerate(unique_values)}
        num_unique_values = len(self._transform_dict)
        if not self.force_valid and num_unique_values == 1:
            msg = (f"all values in column {self.name}, which tends to be string column, "
                   "are the SAME. It'll be excluded since it might be redundant")
            return True, self._make_string_info(None, False, msg)
        if not self.force_valid and num_unique_values == len(flat_arr):
            msg = (f"all values in column {self.name}, which tends to be string column, "
                   "are DIFFERENT. It'll be excluded since it might be redundant")
            return True, self._make_string_info(None, False, msg)
        msg = None
        if num_unique_values >= 1e3:
            msg = f"TOO MANY unique values occurred in column {self.name} ({num_unique_values:^12d})"
        return True, self._make_string_info(flat_arr, True, msg, np.array(unique_values))

    def _check_exclude_categorical(self,
                                   num_samples: int,
                                   num_unique_values: int) -> Tuple[str, Union[str, None]]:
        if not self.force_valid and num_samples == 1:
            msg = (f"all values in column {self.name}, which tends to be categorical column, "
                   "are the SAME. It'll be excluded since it might be redundant")
            return "exclude", msg
        if not self.force_valid and num_samples == num_unique_values:
            msg = (f"all values in column {self.name}, which tends to be categorical column, "
                   "are DIFFERENT. It'll be excluded since it might be redundant")
            return "exclude", msg
        if not self.force_categorical and num_unique_values >= self.numerical_threshold * num_samples:
            msg = (
                f"TOO MANY unique values occurred in column {self.name} ({num_unique_values:^12d}) "
                "which tends to be categorical column, it'll cast to numerical column to save memory "
                "and possibly for better performance"
            )
            return "numerical", msg
        return "keep", None

    def _generate_categorical_transform_dict(self):
        values = self._info.unique_values_sorted_by_counts.tolist()
        transform_dict = {v: i for i, v in enumerate(values) if not math.isnan(v)}
        if self._info.contains_nan:
            transform_dict["nan"] = len(transform_dict)
        self._transform_dict = transform_dict

    def fit(self,
            flat_arr: flat_arr_type) -> "Recognizer":
        is_string, info = self._check_string_column(flat_arr)
        if is_string:
            self._info = info
            return self
        if isinstance(flat_arr[0], (str, np.str_)):
            np_flat = flat_arr_to_float32(flat_arr)
        else:
            np_flat = np.asarray(flat_arr, np.float32)
        nan_mask = np.isnan(np_flat)
        valid_mask = ~nan_mask
        np_flat_valid = np_flat[valid_mask]
        np_flat_valid_int = np_flat_valid.astype(np.int)
        num_samples, num_valid_samples = map(len, [np_flat, np_flat_valid])
        contains_nan = num_samples != num_valid_samples
        if not contains_nan:
            nan_mask = None
        # check whether is_valid or not
        if not self.force_valid and num_valid_samples == 0:
            msg = (f"all values in column {self.name}, which tends to be categorical column, "
                   "are NaN. It'll be excluded since it might be redundant")
            return self._make_invalid_info(msg, contains_nan, nan_mask)
        # check whether it's a numerical column
        all_int = np.allclose(np_flat_valid, np_flat_valid_int)
        if (
            self.force_numerical
            or self.is_label and self.task_type is TaskTypes.REGRESSION
            or not self.force_categorical and not all_int
        ):
            if not (self.is_label and self.task_type is TaskTypes.CLASSIFICATION):
                self._info = FeatureInfo(contains_nan, np_flat, nan_mask=nan_mask)
                return self
        # deal with categorical column
        np_flat_categorical = np_flat_valid_int if all_int else np_flat_valid
        min_feat = np_flat_categorical.min()
        np_flat_categorical -= min_feat
        max_feature_value = np_flat_categorical.max()
        if not all_int or max_feature_value > 1e6:
            unique_values, counts = np.unique(np_flat_categorical, return_counts=True)
            num_unique_values = len(unique_values)
            need_transform = unique_values[-1] != num_unique_values - 1
        else:
            counts = np.bincount(np_flat_categorical)
            unique_values = np.nonzero(counts)[0]
            num_unique_values = len(unique_values)
            need_transform = num_unique_values != len(counts)
            counts = counts[unique_values]
        need_transform = min_feat != 0 or need_transform
        status, msg = self._check_exclude_categorical(num_samples, num_unique_values)
        if status != "keep":
            if status == "numerical":
                self._info = FeatureInfo(contains_nan, np_flat, nan_mask=nan_mask, msg=msg)
                return self
            return self._make_invalid_info(msg, contains_nan, nan_mask)
        sorted_indices = np.argsort(counts)[::-1]
        unique_values = unique_values[sorted_indices] + min_feat
        counter_dict = dict(zip(unique_values, counts))
        if contains_nan:
            unique_values = np.append(unique_values, [float("nan")])
            counter_dict["nan"] = num_samples - num_valid_samples
        need_transform = need_transform or contains_nan
        self._counter = Counter(counter_dict)
        self._info = FeatureInfo(
            contains_nan, np_flat,
            nan_mask=nan_mask,
            need_transform=need_transform,
            column_type=ColumnTypes.CATEGORICAL,
            unique_values_sorted_by_counts=unique_values
        )
        self._generate_categorical_transform_dict()
        return self


__all__ = ["Recognizer"]
