import os
import copy
import dill
import logging

import numpy as np
import datatable as dt

from typing import *
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import timing_context
from cftool.misc import Saving

from .misc import *
from .recognizer import *
from .converters import *
from .processors import *
from ..base import DataBase
from ..types import np_int_type
from ..types import np_float_type


class TabularSplit(NamedTuple):
    split: "TabularData"
    remained: "TabularData"
    split_indices: np.ndarray
    remained_indices: np.ndarray


# TODO : Add outlier detection
class TabularData(DataBase):
    label_idx: int
    label_name: str
    label_type: dt.stype
    column_names: Dict[int, str]

    recognizers: Dict[int, Optional[Recognizer]]
    converters: Dict[int, Optional[Converter]]
    processors: Dict[int, Optional[Processor]]

    def __init__(
        self,
        *,
        simplify: bool = False,
        task_type: task_type_type = TaskTypes.NONE,
        time_series_config: Optional[TimeSeriesConfig] = None,
        label_idx: Optional[int] = None,
        label_name: Optional[str] = None,
        label_type: Optional[dt.stype] = None,
        label_recognizer_config: Optional[Dict[str, Any]] = None,
        label_process_method: Optional[str] = None,
        column_names: Optional[Dict[int, str]] = None,
        valid_columns: Optional[Set[int]] = None,
        invalid_columns: Optional[Set[int]] = None,
        string_columns: Optional[List[int]] = None,
        numerical_columns: Optional[List[int]] = None,
        categorical_columns: Optional[List[int]] = None,
        recognizer_configs: Optional[Dict[int, Dict[str, Any]]] = None,
        process_methods: Optional[Union[str, Dict[int, str]]] = "auto",
        binning_method: str = "fuse",
        default_numerical_process: str = "normalize",
        default_categorical_process: str = "one_hot",
        use_timing_context: bool = True,
        trigger_logging: bool = False,
        verbose_level: int = 1,
    ):
        task_type = parse_task_type(task_type)
        # sanity check
        err_msg = f"{label_type} labels are invalid in {{}} tasks"
        if task_type.is_clf:
            if label_type is not None and is_float(label_type.dtype):
                raise ValueError(err_msg.format("CLASSIFICATION"))
        elif task_type.is_reg:
            if label_type is not None and not is_float(label_type.dtype):
                raise ValueError(err_msg.format("REGRESSION"))
        self._simplify = simplify
        self._task_type = task_type
        self._time_series_config = time_series_config
        self._label_idx = label_idx
        self._label_name = label_name
        self._label_type = label_type
        # column settings
        self._column_names = column_names
        self._valid_columns = valid_columns or set()
        self._invalid_columns = invalid_columns or set()
        self._stypes: Dict[str, dt.stype] = {}
        self._preset_stypes: Dict[int, dt.stype] = {}
        for i in string_columns or []:
            self._preset_stypes[i] = dt.str32
        for i in numerical_columns or []:
            self._preset_stypes[i] = dt.float32
        for i in categorical_columns or []:
            self._preset_stypes[i] = dt.int64
        self._recognizer_configs = recognizer_configs or {}
        self._label_recognizer_config = label_recognizer_config or {}
        self._process_methods = process_methods
        self._binning_method = binning_method
        self._default_numerical_process = default_numerical_process
        self._default_categorical_process = default_categorical_process
        self._label_process_method = label_process_method
        self._is_file = self._is_arr = self._is_np = False
        self._raw_dim: Optional[int] = None
        self._num_classes: Optional[int] = None
        self._x_df: Optional[dt.Frame] = None
        self._y_df: Optional[dt.Frame] = None
        self._raw: Optional[DataTuple] = None
        self._converted: Optional[DataTuple] = None
        self._processed: Optional[DataTuple] = None
        self._timing = use_timing_context
        self._verbose_level = verbose_level
        self._init_logging(verbose_level, trigger=trigger_logging)
        self.excludes: Set[int] = set()

    def __len__(self) -> int:
        processed = self.processed
        if processed is None or processed.x is None:
            return 0
        return len(processed.x)

    def __getitem__(self, indices: np.ndarray) -> data_item_type:
        processed = self.processed
        if processed is None:
            raise ValueError("`processed` is not provided")
        x, y = processed.xy
        assert isinstance(x, np.ndarray)
        y_batch = None if y is None else y[indices]
        return x[indices], y_batch

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TabularData):
            raise NotImplementedError
        if self.raw != other.raw:
            return False
        if self.converted != other.converted:
            return False
        return self.processed == other.processed

    @property
    def is_simplify(self) -> bool:
        return self._simplify

    @property
    def ts_config(self) -> Optional[TimeSeriesConfig]:
        if self._time_series_config is None:
            return None
        id_name = self._time_series_config.id_column_name
        time_name = self._time_series_config.time_column_name
        id_idx = self._time_series_config.id_column_idx
        time_idx = self._time_series_config.time_column_idx
        if id_idx is None:
            if id_name is None:
                msg = "either `id_column_name` or `id_column` should be provided"
                raise ValueError(msg)
            for k, v in self.column_names.items():
                if v == id_name:
                    id_idx = k
                    break
        if id_idx is None:
            raise ValueError(f"id_name '{id_name}' is not found")
        if time_idx is None:
            if time_name is None:
                msg = "either `time_column_name` or `time_column` should be provided"
                raise ValueError(msg)
            for k, v in self.column_names.items():
                if v == time_name:
                    time_idx = k
                    break
        if time_idx is None:
            raise ValueError(f"time_name '{time_name}' is not found")
        raw = self.raw
        if raw is None:
            raise ValueError("`raw` should be provided in `ts_config`")
        raw_xt = raw.xT
        if raw_xt is None:
            raise ValueError("`raw.xT` should be provided in `ts_config`")
        id_column = raw_xt[id_idx]
        time_column = raw_xt[time_idx]
        return TimeSeriesConfig(
            id_name,
            time_name,
            id_idx,
            time_idx,
            id_column,
            time_column,
        )

    @property
    def cache_excludes(self) -> Set[str]:
        return {"recognizers", "converters", "processors", "_x_df", "_y_df"}

    @property
    def data_tuple_base(self) -> Optional[Type]:
        return DataTuple

    @property
    def data_tuple_attributes(self) -> Optional[List[str]]:
        return ["_raw", "_converted", "_processed"]

    @property
    def raw(self) -> Optional[DataTuple]:
        return self._raw

    @property
    def converted(self) -> Optional[DataTuple]:
        return self._converted

    @property
    def processed(self) -> Optional[DataTuple]:
        return self._processed

    @property
    def raw_dim(self) -> int:
        if self._raw_dim is None:
            raise ValueError("`_raw_dim` is not generated yet")
        return self._raw_dim

    @property
    def processed_dim(self) -> int:
        if self._processed is None:
            raise ValueError("`_processed` is not prepared yet")
        if not isinstance(self._processed.x, np.ndarray):
            raise ValueError("`_processed.x` is not prepared yet")
        return self._processed.x.shape[1]

    @property
    def task_type(self) -> TaskTypes:
        if not self._task_type.is_none:
            return self._task_type
        if self.recognizers[-1] is None:
            return TaskTypes.NONE
        self._task_type = TaskTypes.from_column_type(
            self.recognizers[-1].info.column_type,
            is_time_series=self.is_ts,
        )
        return self._task_type

    @property
    def is_clf(self) -> bool:
        return self.task_type.is_clf

    @property
    def is_reg(self) -> bool:
        return self.task_type.is_reg

    @property
    def is_ts(self) -> bool:
        return self.ts_config is not None

    @property
    def is_file(self) -> bool:
        return self._is_file

    @property
    def splitter(self) -> DataSplitter:
        splitter = DataSplitter(time_series_config=self.ts_config, shuffle=False)
        return splitter.fit(self.to_dataset())

    @property
    def ts_indices(self) -> Set[int]:
        ts_config = self.ts_config
        if ts_config is None:
            return set()
        if ts_config.id_column_idx is None:
            raise ValueError("`id_column_idx` not defined in `ts_config`")
        if ts_config.time_column_idx is None:
            raise ValueError("`time_column_idx` not defined in `ts_config`")
        return {ts_config.id_column_idx, ts_config.time_column_idx}

    @property
    def num_classes(self) -> int:
        if self.is_reg:
            return 0
        if self._num_classes is None:
            raise ValueError("`num_classes` is not calculated yet")
        return self._num_classes

    # Core

    @staticmethod
    def _flatten(data: data_type) -> data_type:
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data.ravel()
        flattened: List[List[Any]] = []
        for elem in data:
            flattened.extend(elem)
        return flattened

    def _get_ts_sorting_indices(self) -> None:
        stacked = np.hstack(self.splitter._time_indices_list_in_use)
        self.ts_sorting_indices = stacked[::-1].copy()

    def _inject_label_recognizer(self) -> "Recognizer":
        self._label_recognizer_config["numerical_threshold"] = 1.0
        recognizer = self.recognizers[-1] = Recognizer(
            self.label_name,
            self._is_np,
            is_label=True,
            is_valid=True,
            task_type=self._task_type,
            binning=self._binning_method,
            config=self._label_recognizer_config,
        )
        assert self._raw is not None and self._raw.y is not None
        recognizer.fit(self._y_df, is_preset=False)
        return recognizer

    def _to_simplify_array(self, raw: DataTuple) -> DataTuple:
        if not self._is_file:
            return raw
        x, y = map(np.array, [raw.x, raw.y])
        if self.is_ts:
            ts_config = self.ts_config
            assert ts_config is not None
            pop = [ts_config.id_column_idx, ts_config.time_column_idx]
            x = np.delete(x, pop, axis=1)
        x = x.astype(np_float_type)
        y = y.astype(np_int_type if self.is_clf else np_float_type)
        return DataTuple(x, y)

    def _core_fit(self) -> "TabularData":
        if self._raw is None:
            raise ValueError("`_raw` is not provided")
        if self._raw.x is None:
            raise ValueError("`_raw.x` is not provided")
        self._raw_dim = len(self._raw.x[0])
        if self._simplify:
            self.recognizers = {}
            self.converters = {}
            self.processors = {}
            self._converted = self._processed = self._to_simplify_array(self._raw)
            # fit label recognizer for imbalance sampler
            with timing_context(self, "fit recognizer", enable=self._timing):
                self._inject_label_recognizer()
        else:
            ts_indices = self.ts_indices
            self.recognizers, self.converters = {}, {}
            # convert labels
            if self._raw is None or self._raw.y is None:
                converted_labels = None
                self.recognizers[-1] = None
                self.converters[-1] = None
            else:
                with timing_context(self, "fit recognizer", enable=self._timing):
                    recognizer = self._inject_label_recognizer()
                with timing_context(self, "fit converter", enable=self._timing):
                    converter = Converter.make_with(recognizer)
                    self.converters[-1] = converter
                with timing_context(self, "convert", enable=self._timing):
                    converted_labels = converter.converted_input.reshape([-1, 1])
            # convert features
            converted_features = []
            if self._x_df is None:
                raise ValueError("`_x_df` is required in `_core_fit`")
            for i in range(self.raw_dim):
                column_name = self.column_names[i if i < self.label_idx else i + 1]
                is_valid = None
                if i in self._valid_columns:
                    is_valid = True
                elif i in self._invalid_columns:
                    is_valid = False
                if i == self.raw_dim - 1 == len(self.excludes):
                    if i > 0:
                        self.log_msg(
                            f"last column {column_name} is forced to be valid "
                            "because previous columns are all excluded",
                            self.warning_prefix,
                            verbose_level=2,
                            msg_level=logging.WARNING,
                        )
                    is_valid = True
                with timing_context(self, "fit recognizer", enable=self._timing):
                    recognizer_config = self._recognizer_configs.setdefault(i, {})
                    recognizer = Recognizer(
                        column_name,
                        self._is_np,
                        is_label=False,
                        is_valid=is_valid,
                        task_type=self.task_type,
                        binning=self._binning_method,
                        labels=converted_labels,
                        config=recognizer_config,
                    )
                    recognizer.fit(self._x_df[:, i], is_preset=i in self._preset_stypes)
                    self.recognizers[i] = recognizer
                if not recognizer.info.is_valid:
                    self.log_msg(
                        recognizer.info.msg,
                        self.warning_prefix,
                        2,
                        logging.WARNING,
                    )
                    self.excludes.add(i)
                    continue
                if i not in ts_indices:
                    with timing_context(self, "fit converter", enable=self._timing):
                        converter = Converter.make_with(recognizer)
                        self.converters[i] = converter
                    with timing_context(self, "convert", enable=self._timing):
                        converted = converter.converted_input.astype(np_float_type)
                        converted_features.append(converted)
            converted_x = np.vstack(converted_features).T
            # process features
            self.processors = {}
            processed_features = []
            previous_processors: List[Processor] = []
            idx = 0
            while idx < self.raw_dim:
                if idx in self.excludes or idx in ts_indices:
                    idx += 1
                    continue
                local_converter = self.converters[idx]
                assert local_converter is not None
                column_type = local_converter.info.column_type
                if self._process_methods is None:
                    method = None
                elif isinstance(self._process_methods, str):
                    method = self._process_methods
                else:
                    method = self._process_methods.get(idx, "auto")
                if method is None:
                    method = "identical"
                elif method == "auto":
                    if idx in ts_indices:
                        method = "identical"
                    elif column_type is ColumnTypes.NUMERICAL:
                        method = self._default_numerical_process
                    else:
                        method = self._default_categorical_process
                base = processor_dict[method]
                processor = base.make_with(previous_processors.copy())
                previous_processors.append(processor)
                self.processors[idx] = processor
                columns = converted_x[..., processor.input_indices]
                with timing_context(self, "fit processor", enable=self._timing):
                    processor.fit(columns)
                with timing_context(self, "process", enable=self._timing):
                    processed_features.append(processor.process(columns))
                idx += processor.input_dim
            # process labels
            if converted_labels is None:
                processed_labels = self.processors[-1] = None
            else:
                label_converter = self.converters[-1]
                assert label_converter is not None
                column_type = label_converter.info.column_type
                method = None
                if self._label_process_method is not None:
                    method = self._label_process_method
                if method is None:
                    method = (
                        "normalize"
                        if column_type is ColumnTypes.NUMERICAL
                        else "identical"
                    )
                with timing_context(self, "fit processor", enable=self._timing):
                    processor = processor_dict[method].make_with([])
                    self.processors[-1] = processor.fit(converted_labels)
                with timing_context(self, "process", enable=self._timing):
                    processed_labels = processor.process(converted_labels)
            has_converted_labels = converted_labels is not None
            has_processed_labels = processed_labels is not None
            if self.task_type.is_clf and has_converted_labels and has_processed_labels:
                assert isinstance(converted_labels, np.ndarray)
                assert isinstance(processed_labels, np.ndarray)
                converted_labels = converted_labels.astype(np_int_type)
                processed_labels = processed_labels.astype(np_int_type)
            self._converted = DataTuple(converted_x, converted_labels)
            self._processed = DataTuple(np.hstack(processed_features), processed_labels)
        self.ts_sorting_indices = None
        # time series
        if self.is_ts:
            self._get_ts_sorting_indices()
        # num classes
        if not self.is_reg and self._processed.y is not None:
            assert isinstance(self._processed.y, np.ndarray)
            self._num_classes = self._processed.y.max().item() + 1
        return self

    def _split_df(self, df: dt.Frame) -> Tuple[dt.Frame, dt.Frame, List[int]]:
        x_indices = list(range(df.ncols))
        x_indices.pop(self.label_idx)
        return df[:, x_indices], df[:, self.label_idx], x_indices

    def _read_from_file(self, file_path: str) -> "TabularData":
        self._is_file = True
        # names
        with open(file_path, "r") as f:
            df_head = dt.Frame(f.readline())
        names = list(df_head.names)
        for i, name in (self._column_names or {}).items():
            names[i] = name
        self.column_names = {i: name for i, name in enumerate(names)}
        # get y info
        if self._label_idx is not None:
            while self._label_idx < 0:
                self._label_idx += len(self.column_names) - 1
            self.label_idx = self._label_idx
            label_name = self.column_names[self.label_idx]
            if self._label_name is not None and label_name != self._label_name:
                raise ValueError(
                    f"detected label name ({label_name}) is not identical with "
                    f"the specified label name ({self._label_name})"
                )
        else:
            if self._label_name is None:
                self.label_idx = len(names) - 1
            else:
                try:
                    self.label_idx = names.index(self._label_name)
                except ValueError:
                    raise ValueError(
                        f"specified label name ({self._label_name}) could not be found "
                        f"in the detected names ({names})"
                    )
        self.label_indices = [self.label_idx]
        self.label_name = self.column_names[self.label_idx]
        # stypes
        stypes = {names[i]: v for i, v in self._preset_stypes.items()}
        if self._label_type is not None:
            stypes[self.label_name] = self._label_type
        # frames
        df = dt.Frame(file_path, names=names, stypes=stypes)
        self._x_df, self._y_df, x_indices = self._split_df(df)
        # set names
        self._x_df.names = [self.column_names[i] for i in x_indices]
        self._y_df.names = [self.column_names[self.label_idx]]
        # stypes
        self._stypes = dict(zip(df.names, df.stypes))
        label_type = self._stypes[self.label_name]
        if self._label_type is not None:
            if self._label_type != label_type:
                self.log_msg(
                    "`label_type` will be switched "
                    f"from {self._label_type} to {label_type}",
                    self.warning_prefix,
                )
        self.label_type = label_type
        # core fit
        self._raw = DataTuple.from_dfs(self._x_df, self._y_df)
        return self._core_fit()

    def _read_from_arr(self, x: data_type, y: data_type) -> "TabularData":
        assert x is not None
        self._is_arr = True
        self._is_np = isinstance(x, np.ndarray)
        # check 2d y
        failed = False
        if isinstance(y, list):
            failed = not isinstance(y[0], list)
        elif isinstance(y, np.ndarray):
            failed = len(y.shape) != 2
        if failed:
            raise ValueError("input labels should be 2d")
        # names
        self.label_name = self._label_name or "label"
        x_names = [f"C{i}" for i in range(len(x[0]))]
        if y is None:
            y_dim = 0
            y_names = []
        else:
            y_dim = len(y[0]) if isinstance(y, list) else y.shape[1]
            if y_dim == 1:
                y_names = [self.label_name]
            else:
                y_names = [f"{self.label_name}_{i}" for i in range(y_dim)]
        names = x_names + y_names
        for i, name in (self._column_names or {}).items():
            names[i] = name
        self.column_names = {i: name for i, name in enumerate(names)}
        # x settings
        x_stypes = {x_names[i]: v for i, v in self._preset_stypes.items()}
        x_kwargs = {"names": x_names, "stypes": x_stypes}
        if not self._is_np:
            x_dt = to_dt_data(x)
        else:
            assert isinstance(x, np.ndarray)
            if self._preset_stypes:
                x_dt = x.T.tolist()
            else:
                x_dt = x.astype(np.float32)
                x_stypes = {n: dt.float32 for n in x_names}
                x_kwargs.pop("stypes")
        # y settings
        if self._is_np and self._label_type is not None:
            assert isinstance(y, np.ndarray)
            y = y.astype(self._label_type.dtype)
        y_dt = to_dt_data(y)
        y_kwargs = {"names": y_names}
        all_dim = len(self.column_names)
        self.label_idx = all_dim - y_dim
        self.label_indices = list(range(self.label_idx, all_dim))
        # frames
        self._x_df = dt.Frame(x_dt, **x_kwargs)
        self._y_df = dt.Frame(y_dt, **y_kwargs)
        # stypes
        self._stypes = x_stypes
        self.label_type = self._y_df.stype
        self._stypes[self.label_name] = self.label_type
        # core fit
        self._raw = DataTuple.with_transpose(x, y)
        return self._core_fit()

    def _transform_labels(
        self,
        raw: DataTuple,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        raw_y = raw.y
        if raw_y is None:
            return None, None
        if self._simplify:
            if not isinstance(raw_y, np.ndarray):
                msg = "`TabularData` is set to `simplify` but `raw.y` is not np.ndarray"
                raise ValueError(msg)
            return raw_y, raw_y
        label_converter = self.converters[-1]
        label_processor = self.processors[-1]
        if label_converter is None:
            raise ValueError("label_converter is not generated")
        if label_processor is None:
            raise ValueError("label_processor is not generated")
        converted_labels = label_converter.convert(self._flatten(raw.y))
        converted_labels = converted_labels.reshape([-1, 1])
        transformed_labels = label_processor.process(converted_labels)
        if self.task_type.is_clf:
            converted_labels = converted_labels.astype(np_int_type)
            transformed_labels = transformed_labels.astype(np_int_type)
        return converted_labels, transformed_labels

    def _transform(self, raw: DataTuple) -> Tuple[DataTuple, DataTuple]:
        if self._simplify:
            data_tuple = self._to_simplify_array(raw)
            return data_tuple, data_tuple
        # transform features
        features = raw.xT
        if features is None:
            raise ValueError("`raw` should contain `xT` for TabularData._transform")
        ts_indices = self.ts_indices
        converted_features_list = []
        for i, flat_arr in enumerate(features):
            if i in self.excludes or i in ts_indices:
                continue
            converter = self.converters[i]
            assert converter is not None
            converted_features_list.append(converter.convert(flat_arr))
        converted_features = np.vstack(converted_features_list)
        idx = 0
        processed = []
        while idx < self.raw_dim:
            if idx in self.excludes or idx in ts_indices:
                idx += 1
                continue
            processor = self.processors[idx]
            assert processor is not None
            input_indices = processor.input_indices
            columns = processor.process(converted_features[input_indices].T)
            processed.append(columns)
            idx += processor.input_dim
        transformed_features = np.hstack(processed)
        # transform labels
        converted_labels, transformed_labels = self._transform_labels(raw)
        # aggregate
        transformed = DataTuple(transformed_features, transformed_labels)
        converted = DataTuple(converted_features.T, converted_labels)
        return converted, transformed

    def _dt_kwargs(self, contains_labels: bool) -> Dict[str, Any]:
        stypes = self._stypes.copy()
        names = [self.column_names[i] for i in range(len(self.column_names))]
        if not contains_labels:
            for idx in sorted(self.label_indices)[::-1]:
                names.pop(idx)
            stypes.pop(self.label_name)
        return {"names": names, "stypes": stypes}

    def _get_dfs(
        self,
        x: Union[str, data_type],
        y: data_type = None,
        *,
        contains_labels: bool = True,
    ) -> Tuple[dt.Frame, Optional[dt.Frame]]:
        if isinstance(x, str):
            if not self._is_file:
                raise ValueError("self._is_file is False but file is provided")
            x_df, y_df = self.read_file(x, contains_labels=contains_labels)
        else:
            x_kwargs = self._dt_kwargs(False)
            if not isinstance(x, list):
                x_kwargs.pop("stypes")
            x_df = dt.Frame(to_dt_data(x), **x_kwargs)
            if y is None:
                y_df = None
            else:
                y_names = [self.label_name]
                y_stype = self._stypes[self.label_name]
                y_kwargs = {"names": y_names}
                if isinstance(y, list):
                    y_kwargs["stype"] = y_stype
                y_df = dt.Frame(to_dt_data(y), **y_kwargs)
        return x_df, y_df

    # API

    def read_file(
        self,
        file_path: str,
        *,
        contains_labels: bool = True,
    ) -> Tuple[dt.Frame, Optional[dt.Frame]]:
        df = dt.Frame(file_path, **self._dt_kwargs(contains_labels))
        if not contains_labels:
            if self._raw is not None:
                if df.ncols == self.raw_dim + 1:
                    msg = "file contains labels but 'contains_labels=False' passed in"
                    raise ValueError(msg)
            return df, None
        x_df, y_df, _ = self._split_df(df)
        return x_df, y_df

    def read(
        self,
        x: Union[str, data_type],
        y: Optional[Union[int, data_type]] = None,
        *,
        contains_labels: bool = True,
        **kwargs: Any,
    ) -> "TabularData":
        if isinstance(x, str):
            if y is not None:
                raise ValueError("`y` should not provided when `x` is a file.")
            self._read_from_file(x)
        else:
            if isinstance(y, int):
                y = None
            self._read_from_arr(x, y)
        self.log_timing()
        return self

    def split(self, n: Union[int, float], *, order: str = "auto") -> TabularSplit:
        if order == "auto":
            split = self.splitter.split(n)
            split_indices = split.corresponding_indices
            remained_indices = split.remaining_indices
        else:
            if order not in {"bottom_up", "top_down"}:
                raise NotImplementedError(
                    "`order` should be either 'bottom_up' or "
                    f"'top_down', {order} found"
                )
            if self._raw is None or self._raw.x is None:
                raise ValueError("`_raw.x` is not yet generated")
            num_samples = len(self._raw.x)
            num = n if isinstance(n, int) else int(round(n * num_samples))
            base_indices = np.arange(num_samples)
            if order == "bottom_up":
                split_indices = base_indices[-num:]
                remained_indices = base_indices[:-num]
            else:
                split_indices = base_indices[:num]
                remained_indices = base_indices[num:]
        return self.split_with_indices(split_indices, remained_indices)

    def split_with_indices(
        self,
        split_indices: np.ndarray,
        remained_indices: np.ndarray,
    ) -> TabularSplit:
        raw, converted, processed = self._raw, self._converted, self._processed
        if raw is None:
            raise ValueError("`_raw` data is not generated")
        if converted is None:
            raise ValueError("`_converted` data is not generated")
        if processed is None:
            raise ValueError("`_processed` data is not generated")
        p1: TabularData = copy.copy(self)
        p2: TabularData = copy.copy(self)
        p1._raw, p1._converted, p1._processed = map(
            DataTuple.split_with,
            [raw, converted, processed],
            [split_indices] * 3,
        )
        p2._raw, p2._converted, p2._processed = map(
            DataTuple.split_with,
            [raw, converted, processed],
            [remained_indices] * 3,
        )
        p1.ts_sorting_indices = np.arange(len(p1))
        p2.ts_sorting_indices = np.arange(len(p2))
        return TabularSplit(p1, p2, split_indices, remained_indices)

    def copy_to(
        self,
        x: Union[str, data_type],
        y: data_type = None,
        *,
        contains_labels: bool = True,
    ) -> "TabularData":
        copied = copy.copy(self)
        dfs = self._get_dfs(x, y, contains_labels=contains_labels)
        raw = copied._raw = DataTuple.from_dfs(*dfs)
        converted, copied._processed = self._transform(raw)
        assert isinstance(converted.x, np.ndarray), "internal error occurred"
        if not self._simplify:
            copied_converters: Dict[int, Optional[Converter]] = {
                idx: None if converter is None else copy.copy(converter)
                for idx, converter in self.converters.items()
            }
            label_converter = copied_converters[-1]
            if label_converter is not None:
                label_converter._converted_features = converted.y
            converter_indices = [idx for idx in sorted(copied_converters) if idx != -1]
            for i, idx in enumerate(converter_indices):
                local_converter = copied_converters[idx]
                assert local_converter is not None
                local_converter._converted_features = converted.x[..., i]
            copied.converters = copied_converters
            copied._converted = converted
        if copied.is_ts:
            ts_config = self.ts_config
            assert ts_config is not None
            copied._time_series_config = TimeSeriesConfig(
                ts_config.id_column_name,
                ts_config.time_column_name,
                ts_config.id_column_idx,
                ts_config.time_column_idx,
            )
            copied._get_ts_sorting_indices()
        return copied

    def transform(
        self,
        x: Union[str, data_type],
        y: data_type = None,
        *,
        contains_labels: bool = True,
        return_converted: bool = False,
        **kwargs: Any,
    ) -> Union[DataTuple, Tuple[DataTuple, DataTuple]]:
        dfs = self._get_dfs(x, y, contains_labels=contains_labels)
        bundle = self._transform(DataTuple.from_dfs(*dfs))
        if return_converted:
            return bundle
        return bundle[1]

    def transform_labels(
        self,
        y: data_type,
        *,
        return_converted: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raw = DataTuple(None, y)
        converted_labels, transformed_labels = self._transform_labels(raw)
        if not return_converted:
            return transformed_labels
        return converted_labels, transformed_labels

    def recover_labels(self, y: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        if self._simplify:
            return y
        label_processor = self.processors[-1]
        if label_processor is None:
            raise ValueError("`processor` for labels is not generated")
        label_converter = self.converters[-1]
        assert label_converter is not None
        process_recovered = label_processor.recover(y, inplace=inplace)
        convert_recovered = label_converter.recover(
            process_recovered.ravel(),
            inplace=inplace,
        )
        return convert_recovered.reshape([-1, 1])

    core_folder = "__core__"
    core_file = f"{core_folder}.pkl"
    data_structures_file = "data_structures"

    def save(
        self,
        folder: str,
        *,
        compress: bool = True,
        retain_data: bool = True,
        remove_original: bool = True,
    ) -> "TabularData":
        abs_folder = os.path.abspath(folder)
        base_folder = os.path.dirname(abs_folder)
        core_folder = os.path.join(abs_folder, self.core_folder)
        if retain_data:
            super().save(core_folder, compress=False)
        with lock_manager(base_folder, [folder]):
            if not retain_data:
                Saving.prepare_folder(self, folder)
                instance_dict = shallow_copy_dict(self.__dict__)
                for key in self.cache_excludes:
                    instance_dict.pop(key)
                data_tuple_attributes = self.data_tuple_attributes
                if data_tuple_attributes is not None:
                    for key in data_tuple_attributes:
                        instance_dict.pop(key)
                with open(os.path.join(abs_folder, self.core_file), "wb") as f:
                    dill.dump(instance_dict, f)
            recognizer_dicts: Dict[int, Optional[Recognizer]] = {}
            for idx, recognizer in self.recognizers.items():
                if idx in self.converters:
                    continue
                if recognizer is None:
                    recognizer_dicts[idx] = None
                else:
                    recognizer_dicts[idx] = recognizer.dumps_()
            converter_dicts: Dict[int, Optional[Converter]] = {}
            for idx, converter in self.converters.items():
                if converter is None:
                    converter_dicts[idx] = None
                else:
                    converter_dicts[idx] = converter.dumps_()
            processor_dicts: Dict[int, Optional[Processor]] = {}
            for idx, processor in self.processors.items():
                if processor is None:
                    processor_dicts[idx] = None
                else:
                    processor_dicts[idx] = processor.dumps_()
            with open(os.path.join(abs_folder, self.data_structures_file), "wb") as f:
                dill.dump(
                    {
                        "recognizers": recognizer_dicts,
                        "converters": converter_dicts,
                        "processors": processor_dicts,
                    },
                    f,
                )
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        return self

    @classmethod
    def load(
        cls,
        folder: str,
        *,
        compress: bool = True,
        verbose_level: int = 0,
    ) -> "TabularData":
        data = cls(verbose_level=verbose_level)
        abs_folder = os.path.abspath(folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [folder]):
            with Saving.compress_loader(
                folder,
                compress,
                remove_extracted=True,
                logging_mixin=data,
            ):
                # core
                retain_data = True
                core_file = os.path.join(abs_folder, cls.core_file)
                if os.path.isfile(core_file):
                    retain_data = False
                    with open(core_file, "rb") as f:
                        data.__dict__.update(dill.load(f))
                else:
                    core_folder = os.path.join(abs_folder, cls.core_folder)
                    with data._data_tuple_context(is_saving=False):
                        Saving.load_instance(
                            data,
                            core_folder,
                            log_method=data.log_msg,
                        )
                recognizers: Dict[int, Optional[Recognizer]] = {}
                converters: Dict[int, Optional[Converter]] = {}
                processors: Dict[int, Optional[Processor]] = {}
                # data structures
                ds_path = os.path.join(abs_folder, cls.data_structures_file)
                with open(ds_path, "rb") as f:
                    data_structures = dill.load(f)
                # converters & corresponding recognizers
                converters_dicts = data_structures["converters"]
                for idx, converter_dict_ in converters_dicts.items():
                    converter = converters[idx] = Converter.loads(converter_dict_)
                    recognizers[idx] = converter._recognizer
                # other recognizers
                recognizers_dicts = data_structures["recognizers"]
                for idx, recognizer_dict_ in recognizers_dicts.items():
                    recognizers[idx] = Recognizer.loads(recognizer_dict_)
                # processors
                if not data.is_simplify:
                    previous_processors: List[Processor] = []
                    processors_dicts = data_structures["processors"]
                    label_processor_data = processors_dicts.pop(-1)
                    if label_processor_data is None:
                        recognizers[-1] = None
                        converters[-1] = None
                        processors[-1] = None
                    else:
                        processors[-1] = Processor.loads(
                            label_processor_data,
                            previous_processors=[],
                        )
                    for idx in sorted(processors_dicts):
                        processor = processors[idx] = Processor.loads(
                            processors_dicts[idx],
                            previous_processors=previous_processors.copy(),
                        )
                        previous_processors.append(processor)
                # assign
                data.recognizers = recognizers
                data.converters = converters
                data.processors = processors
                # data
                if not retain_data:
                    data._raw = data._converted = data._processed = None
                else:
                    msg = (
                        "data file corrupted, "
                        "this may cause by backward compatibility breaking"
                    )
                    if data._converted is None:
                        raise ValueError(msg)
                    if not isinstance(data._converted.x, np.ndarray):
                        raise ValueError(msg)
                    if not data._simplify:
                        converted_features = data._converted.x
                        indices = [idx for idx in sorted(converters) if idx != -1]
                        for i, idx in enumerate(indices):
                            converter_ = converters[idx]
                            assert converter_ is not None
                            converter_._converted_features = converted_features[..., i]
                        label_converter = converters[-1]
                        if label_converter is not None:
                            if not isinstance(data._converted.y, np.ndarray):
                                raise ValueError(msg)
                            y_flatten = data._converted.y.flatten()
                            label_converter._converted_features = y_flatten
        return data

    def to_dataset(self) -> TabularDataset:
        processed = self.processed
        if processed is None:
            raise ValueError("`processed` is not provided")
        return TabularDataset(*processed.xy, task_type=self.task_type)

    @classmethod
    def from_dataset(cls, dataset: TabularDataset, **kwargs: Any) -> "TabularData":
        task_type = kwargs.pop("task_type", dataset.task_type)
        return cls(task_type=task_type, **kwargs).read(*dataset.xy)

    @classmethod
    def simple(
        cls,
        task_type: task_type_type,
        *,
        simplify: bool = False,
        **kwargs: Any,
    ) -> "TabularData":
        if simplify:
            kwargs["simplify"] = simplify
        else:
            kwargs.setdefault("binning_method", "identical")
            kwargs.setdefault("default_numerical_process", "identical")
            kwargs.setdefault("default_categorical_process", "identical")
        return cls(task_type=task_type, verbose_level=0, **kwargs)


__all__ = ["TabularData"]
