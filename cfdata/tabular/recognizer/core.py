import math

import numpy as np
import datatable as dt

from typing import *
from collections import Counter
from cftool.misc import shallow_copy_dict
from cftool.misc import get_counter_from_arr

from ..misc import *
from ...types import *


class Recognizer(DataStructure):
    df: dt.Frame
    info: FeatureInfo
    counter: Counter
    transform_dict: transform_dict_type
    num_unique_bound: Optional[int]

    def __init__(
        self,
        name: str,
        is_np: bool,
        *,
        is_label: bool = False,
        is_valid: Optional[bool] = None,
        task_type: task_type_type = TaskTypes.NONE,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.is_np = is_np
        self.is_label = is_label
        self.is_valid = is_valid
        self.task_type = parse_task_type(task_type)
        self._init_config(config)

    def __str__(self) -> str:
        return f"Recognizer({self.info.column_type})"

    __repr__ = __str__

    @property
    def num_unique_values(self) -> Union[int, float]:
        if self.info.is_numerical:
            return math.inf
        return len(self._transformed_unique_values)

    def _init_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = {}
        self.config = config
        self._num_thresh = config.setdefault("numerical_threshold", 0.5)
        default_bound = 128
        self.num_unique_bound = config.setdefault("num_unique_bound", default_bound)
        self._truncate_ratio = config.setdefault("truncate_ratio", 0.99)
        if self.num_unique_bound is None:
            default_fuse_threshold = 1.0 / default_bound
        else:
            default_fuse_threshold = 1.0 / self.num_unique_bound
        self._fuse_thresh = config.setdefault("fuse_threshold", default_fuse_threshold)
        if self.num_unique_bound is None:
            default_fuse_fix = int(default_bound // 2)
        else:
            default_fuse_fix = int(round(0.5 * self.num_unique_bound))
        self._num_fuse_fix = config.setdefault("num_fuse_fix", default_fuse_fix)

    def _make_invalid_info(
        self,
        msg: Optional[str],
        contains_nan: bool,
        nan_mask: np.ndarray,
    ) -> "Recognizer":
        self.info = FeatureInfo(
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
            num_unique_bound=self.num_unique_bound,
            unique_values_sorted_by_counts=unique_values,
            sorted_counts=sorted_counts,
        )

    def _make_dummy_info(
        self,
        unique_values: np.ndarray,
        sorted_counts: np.ndarray,
    ) -> FeatureInfo:
        return FeatureInfo(
            None,
            None,
            num_unique_bound=self.num_unique_bound,
            unique_values_sorted_by_counts=unique_values,
            sorted_counts=sorted_counts,
        )

    def _get_transform_dict(
        self,
        check_nan: bool,
        values: Union[List[str], List[float]],
        sorted_counts: np.ndarray,
        info: Optional[FeatureInfo] = None,
    ) -> Tuple[transform_dict_type, List[int]]:
        if info is None:
            info = self._make_dummy_info(np.array(values), sorted_counts)

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
        df: dt.Frame,
        np_flat: np.ndarray,
        nan_mask: np.ndarray,
    ) -> Tuple[bool, Optional[FeatureInfo]]:
        if not is_string(df.stype.dtype):
            return False, None
        msg: Optional[str] = None
        np_flat[nan_mask] = "nan"
        self.counter = get_counter_from_arr(np_flat)
        most_common = sorted(
            self.counter.most_common(),
            key=lambda pair: pair[::-1],
            reverse=True,
        )
        unique_values = [elem[0] for elem in most_common]
        sorted_counts = np.array([elem[1] for elem in most_common])
        num_unique_values = len(unique_values)
        if not self.is_valid and num_unique_values == 1:
            msg = (
                f"all values in column {self.name}, which tends to be string column, "
                "are the SAME. It'll be excluded since it might be redundant"
            )
            return True, self._make_string_info(None, False, msg)
        if not self.is_valid and num_unique_values == len(np_flat):
            msg = (
                f"all values in column {self.name}, which tends to be string column, "
                "are DIFFERENT. It'll be excluded since it might be redundant"
            )
            return True, self._make_string_info(None, False, msg)
        unique_ratio = num_unique_values / len(np_flat)
        if not self.is_valid and unique_ratio >= 0.5:
            msg = (
                f"unique values occurred in column {self.name}, which tends to be "
                f"string column, are TOO MANY (ratio={unique_ratio:8.6f}). "
                "It'll be excluded since it might be redundant"
            )
            return True, self._make_string_info(None, False, msg)
        pack = self._get_transform_dict(False, unique_values, sorted_counts)
        self.transform_dict, self._transformed_unique_values = pack
        return True, self._make_string_info(
            np_flat,
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
        if num_unique_values >= self._num_thresh * num_samples:
            msg = (
                f"TOO MANY unique values occurred in column {self.name} "
                f"({num_unique_values:^12d}) which tends to be categorical column, "
                "it'll cast to numerical column to save memory "
                "and possibly for better performance"
            )
            return "numerical", msg
        return "keep", None

    def _generate_categorical_transform_dict(self) -> None:
        unique_values = self.info.unique_values_sorted_by_counts
        sorted_counts = self.info.sorted_counts
        assert unique_values is not None and sorted_counts is not None
        values = list(map(float, unique_values.tolist()))
        pack = self._get_transform_dict(True, values, sorted_counts, self.info)
        transform_dict, self._transformed_unique_values = pack
        if self.info.contains_nan:
            num_transformed_unique = len(self._transformed_unique_values)
            transform_dict["nan"] = num_transformed_unique
            self._transformed_unique_values.append(num_transformed_unique)
        self.transform_dict = transform_dict

    def fit(self, df: dt.Frame, *, is_preset: bool) -> "Recognizer":
        if self.is_label and is_preset:
            raise ValueError("`is_preset` should always be False when `is_label`")
        msg: Optional[str]
        dtype = df.stype.dtype
        # check name
        if len(df.names) != 1 or df.names[0] != self.name:
            raise ValueError(
                f"df.name ({df.names[0]}) is not identical with "
                f"Recognizer.name ({self.name})"
            )
        # check valid
        if self.is_valid is False:
            self.info = FeatureInfo(
                None,
                None,
                False,
                msg=f"current column ({self.name}) is forced to be invalid",
            )
            return self
        # check string
        df_data = df.to_numpy()
        if isinstance(df_data, np.ma.core.MaskedArray):
            np_flat = df_data.data.ravel()
            nan_mask = df_data.mask.ravel()
        else:
            np_flat = df_data.ravel()
            nan_mask = np.zeros_like(np_flat, dtype=np.bool)
        is_string_, info = self._check_string_column(df, np_flat, nan_mask)
        if is_string_:
            assert info is not None
            self.info = info
            return self
        # check bool
        if is_bool(dtype):
            dtype = np.int64
            np_flat = np_flat.astype(dtype)
        # gather info
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
            if self.is_label:
                raise ValueError("label column is detected as all-invalid")
            if self.is_valid:
                np_flat = np.zeros_like(np_flat)
                self.info = FeatureInfo(
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
        like_int = np.allclose(np_flat_valid, np_flat_valid_int)
        not_int = is_float(dtype) and (is_preset or not like_int)
        is_reg_label = self.is_label and self.task_type.is_reg
        if not_int or is_reg_label:
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
            self.info = FeatureInfo(
                contains_nan,
                np_flat.astype(np_float_type),
                nan_mask=nan_mask,
                is_valid=is_valid,
                msg=msg,
            )
            return self
        # deal with categorical column
        min_feat = np_flat_valid_int.min()
        np_flat_valid_int -= min_feat
        max_feature_value = np_flat_valid_int.max()
        if max_feature_value > 1e6:
            unique_values, counts = np.unique(np_flat_valid_int, return_counts=True)
            num_unique_values = len(unique_values)
            need_transform = unique_values[-1] != num_unique_values - 1
        else:
            counts = np.bincount(np_flat_valid_int)
            unique_values = np.nonzero(counts)[0]
            num_unique_values = len(unique_values)
            need_transform = num_unique_values != len(counts)
            counts = counts[unique_values]
        need_transform = min_feat != 0 or need_transform
        status, msg = self._check_exclude_categorical(num_samples, num_unique_values)
        if not self.is_label and status != "keep":
            if status == "numerical":
                self.info = FeatureInfo(
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
            # we need to go back to float32 to inject nan so we can
            #  correctly transform categorical columns into 'ordered' indices later on
            np_flat = np_flat.copy().astype(np.float32)
            np_flat[nan_mask] = np.nan
        need_transform = need_transform or contains_nan
        self.counter = Counter(counter_dict)
        self.info = FeatureInfo(
            contains_nan,
            np_flat,
            nan_mask=nan_mask,
            need_transform=need_transform,
            column_type=ColumnTypes.CATEGORICAL,
            num_unique_bound=self.num_unique_bound,
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
        recognizer = cls("", False)
        recognizer.__dict__.update(instance_dict)
        return recognizer


__all__ = ["Recognizer"]
