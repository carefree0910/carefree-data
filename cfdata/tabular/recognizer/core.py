import math

import numpy as np

from typing import *
from collections import Counter
from cftool.misc import shallow_copy_dict
from cftool.misc import get_counter_from_arr

from ..misc import *
from ...types import *
from ...misc.c import *


class Recognizer(DataStructure):
    def __init__(
        self,
        column_name: str,
        *,
        is_label: bool = False,
        task_type: TaskTypes = TaskTypes.NONE,
        is_valid: Optional[bool] = None,
        is_string: Optional[bool] = None,
        is_numerical: Optional[bool] = None,
        is_categorical: Optional[bool] = None,
        numerical_threshold: float = 0.5,
    ):
        # is_* :
        # - `None` means no information
        # - `False` means 'force not to *', `True` means 'force to *'
        self.name = column_name
        self.is_label = is_label
        self.task_type = task_type
        if is_string is False and is_numerical is False and is_categorical is False:
            if is_valid is None:
                is_valid = False
            elif is_valid:
                raise ValueError(
                    f"column '{column_name}' is neither string, numerical nor "
                    "categorical, but it is still set to be valid"
                )
        self.is_valid = is_valid
        self.is_string = is_string
        self.is_numerical = is_numerical
        self.is_categorical = is_categorical
        self.numerical_threshold = numerical_threshold
        self._info: FeatureInfo
        self._counter: Counter
        self._transform_dict: Dict[Union[str, int], int]

    def __str__(self) -> str:
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

    @property
    def num_unique_values(self) -> Union[int, float]:
        if self._info.is_numerical:
            return math.inf
        return len(self._transform_dict)

    def _make_invalid_info(
        self,
        msg: Optional[str],
        contains_nan: bool,
        nan_mask: np.ndarray,
    ) -> "Recognizer":
        self._info = FeatureInfo(
            contains_nan,
            None,
            nan_mask=nan_mask,
            is_valid=False,
            msg=msg,
        )
        return self

    @staticmethod
    def _make_string_info(
        flat_arr: np.ndarray,
        is_valid: bool,
        msg: Optional[str],
        unique_values: Optional[np.ndarray] = None,
    ) -> FeatureInfo:
        return FeatureInfo(
            False,
            flat_arr,
            is_valid=is_valid,
            msg=msg,
            need_transform=True,
            column_type=ColumnTypes.STRING,
            unique_values_sorted_by_counts=unique_values,
        )

    def _check_string_column(
        self,
        flat_arr: flat_arr_type,
    ) -> Tuple[bool, Optional[FeatureInfo]]:
        if self.is_numerical or self.is_categorical or self.is_string is False:
            return False, None
        msg: Optional[str] = None
        is_reg_label = self.is_label and self.task_type.is_reg
        if self.is_string:
            if is_reg_label:
                raise ValueError("task_type is REGRESSION but labels are set to string")
        else:
            all_numeric = is_all_numeric(flat_arr)
            if is_reg_label and not all_numeric:
                msg = "task_type is REGRESSION but labels are not all numeric"
                raise ValueError(msg)
            if all_numeric:
                return False, None
        self._counter = get_counter_from_arr(flat_arr)
        unique_values = [elem[0] for elem in self._counter.most_common()]
        self._transform_dict = {v: i for i, v in enumerate(unique_values)}
        num_unique_values = len(self._transform_dict)
        if not self.is_valid and num_unique_values == 1:
            msg = (
                f"all values in column {self.name}, which tends to be string column, "
                "are the SAME. It'll be excluded since it might be redundant"
            )
            return True, self._make_string_info(None, False, msg)
        if not self.is_valid and num_unique_values == len(flat_arr):
            msg = (
                f"all values in column {self.name}, which tends to be string column, "
                "are DIFFERENT. It'll be excluded since it might be redundant"
            )
            return True, self._make_string_info(None, False, msg)
        if num_unique_values >= 1e3:
            msg = (
                f"TOO MANY unique values occurred in column {self.name} "
                f"({num_unique_values:^12d})"
            )
        return True, self._make_string_info(
            flat_arr, True, msg, np.array(unique_values)
        )

    def _check_exclude_categorical(
        self,
        num_samples: int,
        num_unique_values: int,
    ) -> Tuple[str, Optional[str]]:
        if not self.is_valid and num_unique_values == 1:
            msg = (
                f"all values in column {self.name}, which tends to be "
                "categorical column, are the SAME. It'll be excluded "
                "since it might be redundant"
            )
            return "exclude", msg
        if not self.is_valid and num_samples == num_unique_values:
            msg = (
                f"all values in column {self.name}, which tends to be "
                "categorical column, are DIFFERENT. It'll be excluded "
                "since it might be redundant"
            )
            return "exclude", msg
        if (
            not self.is_categorical
            and num_unique_values >= self.numerical_threshold * num_samples
        ):
            msg = (
                f"TOO MANY unique values occurred in column {self.name} "
                f"({num_unique_values:^12d}) which tends to be categorical column, "
                "it'll cast to numerical column to save memory "
                "and possibly for better performance"
            )
            return "numerical", msg
        return "keep", None

    def _generate_categorical_transform_dict(self) -> None:
        unique_values = self._info.unique_values_sorted_by_counts
        assert unique_values is not None
        values = unique_values.tolist()
        transform_dict = {v: i for i, v in enumerate(values) if not math.isnan(v)}
        if self._info.contains_nan:
            transform_dict["nan"] = len(transform_dict)
        self._transform_dict = transform_dict

    def fit(self, flat_arr: flat_arr_type) -> "Recognizer":
        msg: Optional[str]
        if self.is_valid is False:
            self._info = FeatureInfo(
                None,
                None,
                False,
                msg=f"current column ({self.name}) is forced to be invalid",
            )
            return self
        is_string, info = self._check_string_column(flat_arr)
        if is_string:
            assert info is not None
            self._info = info
            return self
        if isinstance(flat_arr[0], (str, np.str_)):
            np_flat = flat_arr_to_float32(flat_arr)
            if np_float_type != np.float32:
                np_flat = np.asarray(np_flat, np_float_type)
        else:
            np_flat = np.asarray(flat_arr, np_float_type)
        nan_mask = np.isnan(np_flat)
        valid_mask = ~nan_mask
        np_flat_valid = np_flat[valid_mask]
        np_flat_valid_int = np_flat_valid.astype(np_int_type)
        num_samples, num_valid_samples = map(len, [np_flat, np_flat_valid])
        contains_nan = num_samples != num_valid_samples
        if not contains_nan:
            nan_mask = None
        # check whether all nan or not
        if num_valid_samples == 0:
            if self.is_valid:
                np_flat = np.zeros_like(np_flat)
                self._info = FeatureInfo(
                    contains_nan,
                    np_flat,
                    nan_mask=np.zeros_like(nan_mask),
                )
                return self
            msg = (
                f"all values in column {self.name}, which tends to be "
                "categorical column, are NaN. It'll be excluded since "
                "it might be redundant"
            )
            return self._make_invalid_info(msg, contains_nan, nan_mask)
        # check whether it's a numerical column
        all_int = np.allclose(np_flat_valid, np_flat_valid_int)
        is_classification_label = self.is_label and self.task_type.is_clf
        if (
            self.is_numerical
            or self.is_numerical is None
            and (
                self.is_categorical is False
                or self.is_label
                and self.task_type.is_reg
                or self.is_categorical is None
                and not all_int
            )
        ):
            if not is_classification_label:
                msg, is_valid = None, True
                if (
                    self.is_valid is None
                    and np_flat_valid.max() - np_flat_valid.min() <= 1e-8
                ):
                    is_valid = False
                    msg = (
                        f"all values in column {self.name}, which tends to be "
                        "numerical column, are ALL CLOSE. It'll be excluded since "
                        "it might be redundant"
                    )
                self._info = FeatureInfo(
                    contains_nan,
                    np_flat,
                    nan_mask=nan_mask,
                    is_valid=is_valid,
                    msg=msg,
                )
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
        if not is_classification_label and status != "keep":
            if status == "numerical":
                self._info = FeatureInfo(
                    contains_nan,
                    np_flat,
                    nan_mask=nan_mask,
                    msg=msg,
                )
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
            contains_nan,
            np_flat,
            nan_mask=nan_mask,
            need_transform=need_transform,
            column_type=ColumnTypes.CATEGORICAL,
            unique_values_sorted_by_counts=unique_values,
        )
        self._generate_categorical_transform_dict()
        return self

    def dumps_(self) -> Any:
        instance_dict = shallow_copy_dict(self.__dict__)
        instance_dict["_info"] = FeatureInfo(
            self.info.contains_nan,
            None,
            self.info.is_valid,
            None,
            self.info.need_transform,
            self.info.column_type,
            self.info.unique_values_sorted_by_counts,
            self.info.msg,
        )
        return instance_dict

    @classmethod
    def loads(cls, instance_dict: Dict[str, Any], **kwargs: Any) -> "Recognizer":
        recognizer = cls("")
        recognizer.__dict__.update(instance_dict)
        return recognizer


__all__ = ["Recognizer"]
