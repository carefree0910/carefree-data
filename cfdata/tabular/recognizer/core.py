import math

import numpy as np
import datatable as dt

from typing import *
from collections import Counter
from cftool.misc import shallow_copy_dict
from cftool.misc import get_counter_from_arr

from ..misc import *
from ...types import *
from .binning import BinningBase
from .binning import BinningError


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
        binning: str = "auto",
        is_label: bool = False,
        is_valid: Optional[bool] = None,
        task_type: task_type_type = TaskTypes.NONE,
        labels: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.is_np = is_np
        self.binning = binning
        self.labels = labels
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
        default_bound = config.setdefault("default_bound", 128)
        self.num_unique_bound = config.setdefault("num_unique_bound", default_bound)

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
        np_flat: np.ndarray,
        unique_values: np.ndarray,
        sorted_counts: np.ndarray,
    ) -> FeatureInfo:
        return FeatureInfo(
            None,
            np_flat,
            num_unique_bound=self.num_unique_bound,
            unique_values_sorted_by_counts=unique_values,
            sorted_counts=sorted_counts,
        )

    def _get_transform_dict(
        self,
        info: FeatureInfo,
        check_nan: bool,
        unique_values: Union[List[str], List[float]],
        sorted_counts: np.ndarray,
    ) -> Tuple[transform_dict_type, List[int]]:
        def _core(
            unique_values_: Union[List[str], List[float]],
            indices: Optional[List[int]] = None,
        ) -> transform_dict_type:
            if indices is None:
                indices = list(range(len(unique_values_)))
            iterator = zip(indices, unique_values_)
            td: transform_dict_type = {}
            if not check_nan:
                for i, v in iterator:
                    assert isinstance(v, (str, float))
                    td[v] = i
                return td
            for i, v in iterator:
                assert isinstance(v, float)
                if math.isnan(v):
                    continue
                td[v] = i
            return td

        if self.is_label:
            return _core(unique_values), list(range(len(unique_values)))

        if self.binning != "auto":
            binning_type = self.binning
        else:
            binning_type = "opt" if self.labels is not None else "fuse"
        args = binning_type, self.labels, self.task_type, self.config
        binning = BinningBase.make(*args)
        results = binning.binning(info, sorted_counts, unique_values)
        fused_indices, unique_values, transformed_unique_values = results
        return _core(unique_values, fused_indices), transformed_unique_values

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
        unique_values_arr = np.array(unique_values)
        dummy_info = self._make_dummy_info(np_flat, unique_values_arr, sorted_counts)
        args = dummy_info, False, unique_values, sorted_counts
        try:
            pack = self._get_transform_dict(*args)
        except BinningError:
            self.binning = "fuse"
            pack = self._get_transform_dict(*args)
        except Exception as err:
            msg = (
                f"column {self.name}, which tends to be string column, "
                f"cannot be transformed ({err})."
            )
            if not self.is_valid:
                msg = f"{msg} It'll be excluded since it might be redundant"
                return True, self._make_string_info(None, False, msg)
            self.binning = "fuse"
            pack = self._get_transform_dict(*args)
        self.transform_dict, self._transformed_unique_values = pack
        return True, self._make_string_info(
            np_flat,
            True,
            msg,
            unique_values_arr,
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
        pack = self._get_transform_dict(self.info, True, values, sorted_counts)
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
        if len(df.names) == 1:
            if df.names[0] != self.name:
                raise ValueError(
                    f"df.name ({df.names[0]}) is not identical with "
                    f"Recognizer.name ({self.name})"
                )
        else:
            for name in df.names:
                if not name.startswith(self.name):
                    raise ValueError(
                        f"one name ({name}) in df.names does not start with "
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
        try:
            self._generate_categorical_transform_dict()
        except BinningError:
            self.binning = "fuse"
            self._generate_categorical_transform_dict()
        except Exception as err:
            msg = (
                f"column {self.name}, which tends to be categorical column, "
                f"cannot be transformed ({err})."
            )
            if not self.is_valid:
                msg = f"{msg} It'll be excluded since it might be redundant"
                return self._make_invalid_info(msg, contains_nan, nan_mask)
            self.binning = "fuse"
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
