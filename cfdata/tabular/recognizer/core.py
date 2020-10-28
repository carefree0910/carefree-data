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
        task_type: task_type_type = TaskTypes.NONE,
        is_valid: Optional[bool] = None,
        is_string: Optional[bool] = None,
        is_numerical: Optional[bool] = None,
        is_categorical: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # is_* :
        # - `None` means no information
        # - `False` means 'force not to *', `True` means 'force to *'
        self.name = column_name
        self.is_label = is_label
        self.task_type = parse_task_type(task_type)
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
        self._num_unique_bound: Optional[int]
        self._init_config(config)
        self._info: FeatureInfo
        self._counter: Counter
        self._transform_dict: transform_dict_type

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
    def transform_dict(self) -> transform_dict_type:
        return self._transform_dict

    @property
    def num_unique_values(self) -> Union[int, float]:
        if self._info.is_numerical:
            return math.inf
        return len(self._transformed_unique_values)

    def _init_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = {}
        self.config = config
        self._num_thresh = config.setdefault("numerical_threshold", 0.5)
        default_bound = 128
        self._num_unique_bound = config.setdefault("num_unique_bound", default_bound)
        self._truncate_ratio = config.setdefault("truncate_ratio", 0.99)
        if self._num_unique_bound is None:
            default_fuse_threshold = 1.0 / default_bound
        else:
            default_fuse_threshold = 1.0 / self._num_unique_bound
        self._fuse_thresh = config.setdefault("fuse_threshold", default_fuse_threshold)
        if self._num_unique_bound is None:
            default_fuse_fix = int(default_bound // 2)
        else:
            default_fuse_fix = int(round(0.5 * self._num_unique_bound))
        self._num_fuse_fix = config.setdefault("num_fuse_fix", default_fuse_fix)

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

    def _make_string_info(
        self,
        flat_arr: Optional[np.ndarray],
        is_valid: bool,
        msg: Optional[str],
        unique_values: Optional[np.ndarray] = None,
        sorted_counts: Optional[np.ndarray] = None,
    ) -> FeatureInfo:
        return FeatureInfo(
            False,
            flat_arr,
            msg=msg,
            is_valid=is_valid,
            need_transform=True,
            column_type=ColumnTypes.STRING,
            num_unique_bound=self._num_unique_bound,
            unique_values_sorted_by_counts=unique_values,
            sorted_counts=sorted_counts,
        )

    def _make_dummy_info(
        self,
        contains_nan: Optional[bool],
        unique_values: np.ndarray,
        sorted_counts: np.ndarray,
    ) -> FeatureInfo:
        return FeatureInfo(
            contains_nan,
            None,
            num_unique_bound=self._num_unique_bound,
            unique_values_sorted_by_counts=unique_values,
            sorted_counts=sorted_counts,
        )

    def _get_transform_dict(
        self,
        check_nan: bool,
        values: Union[List[str], List[float]],
        sorted_counts: np.ndarray,
        info: Optional[FeatureInfo] = None,
        contains_nan: Optional[bool] = None,
    ) -> Tuple[transform_dict_type, List[int]]:
        if info is None:
            info = self._make_dummy_info(
                contains_nan,
                np.array(values),
                sorted_counts,
            )

        def _core(
            values_: Union[List[str], List[float]],
            indices: Optional[List[int]] = None,
        ) -> transform_dict_type:
            if indices is None:
                indices = list(range(len(values_)))
            iterator = zip(indices, values_)
            td: transform_dict_type = {}
            if not check_nan:
                for i_, v_ in iterator:
                    assert isinstance(v_, (str, float))
                    td[v_] = i_
                return td
            for i_, v_ in iterator:
                assert isinstance(v_, float)
                if math.isnan(v_):
                    continue
                td[v_] = i_
            return td

        if not info.need_truncate:
            transformed_unique_values = list(range(len(values)))
            return _core(values), transformed_unique_values
        # truncate
        counts_cumsum = np.cumsum(sorted_counts)
        counts_cumsum_ratio = counts_cumsum / counts_cumsum[-1]
        truncate_mask = counts_cumsum_ratio >= self._truncate_ratio
        truncate_idx = np.nonzero(truncate_mask)[0][0]
        values = values[: truncate_idx + 1]
        # fuse
        idx = 0
        cumulate = 0.0
        fused_indices = []
        for i, ratio in enumerate(counts_cumsum_ratio[: truncate_idx + 1]):
            fused_indices.append(idx)
            if i < self._num_fuse_fix or ratio >= self._fuse_thresh + cumulate:
                idx += 1
                cumulate = ratio
        transformed_unique_values = sorted(set(fused_indices))
        return _core(values, fused_indices), transformed_unique_values

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
        most_common = self._counter.most_common()
        unique_values = [elem[0] for elem in most_common]
        sorted_counts = np.array([elem[1] for elem in most_common])
        num_unique_values = len(unique_values)
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
        pack = self._get_transform_dict(False, unique_values, sorted_counts)
        self._transform_dict, self._transformed_unique_values = pack
        return True, self._make_string_info(
            flat_arr,
            True,
            msg,
            np.array(unique_values),
            sorted_counts,
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
            and num_unique_values >= self._num_thresh * num_samples
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
        sorted_counts = self._info.sorted_counts
        assert unique_values is not None and sorted_counts is not None
        values = list(map(float, unique_values.tolist()))
        pack = self._get_transform_dict(True, values, sorted_counts, self._info)
        transform_dict, self._transformed_unique_values = pack
        if self._info.contains_nan:
            num_transformed_unique = len(self._transformed_unique_values)
            transform_dict["nan"] = num_transformed_unique
            self._transformed_unique_values.append(num_transformed_unique)
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
        num_nan = num_samples - num_valid_samples
        contains_nan = num_nan != 0
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
        sorted_counts = counts[sorted_indices].astype(np_float_type)
        counter_dict = dict(zip(unique_values, counts))
        if contains_nan:
            unique_values = np.append(unique_values, [float("nan")])
            sorted_counts = np.append(sorted_counts, [float(num_nan)])
            counter_dict["nan"] = num_nan
        need_transform = need_transform or contains_nan
        self._counter = Counter(counter_dict)
        self._info = FeatureInfo(
            contains_nan,
            np_flat,
            nan_mask=nan_mask,
            need_transform=need_transform,
            column_type=ColumnTypes.CATEGORICAL,
            num_unique_bound=self._num_unique_bound,
            unique_values_sorted_by_counts=unique_values,
            sorted_counts=sorted_counts,
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
            self.info.num_unique_bound,
            self.info.unique_values_sorted_by_counts,
            self.info.sorted_counts,
            self.info.msg,
        )
        return instance_dict

    @classmethod
    def loads(cls, instance_dict: Dict[str, Any], **kwargs: Any) -> "Recognizer":
        recognizer = cls("")
        recognizer.__dict__.update(instance_dict)
        return recognizer


__all__ = ["Recognizer"]
