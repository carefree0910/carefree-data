import os
import copy
import dill
import logging

import numpy as np

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


class TabularSplit(NamedTuple):
    split: "TabularData"
    remained: "TabularData"
    split_indices: np.ndarray
    remained_indices: np.ndarray


# TODO : Add outlier detection
class TabularData(DataBase):
    def __init__(
        self,
        *,
        simplify: bool = False,
        task_type: TaskTypes = TaskTypes.NONE,
        time_series_config: Optional[TimeSeriesConfig] = None,
        label_name: Optional[str] = None,
        string_label: Optional[bool] = None,
        numerical_label: Optional[bool] = None,
        categorical_label: Optional[bool] = None,
        column_names: Optional[Dict[int, str]] = None,
        valid_columns: Optional[List[int]] = None,
        string_columns: Optional[List[int]] = None,
        numerical_columns: Optional[List[int]] = None,
        categorical_columns: Optional[List[int]] = None,
        process_methods: Optional[Union[str, Dict[int, str]]] = "auto",
        default_numerical_process: str = "normalize",
        default_categorical_process: str = "one_hot",
        label_process_method: Optional[str] = None,
        numerical_threshold: Optional[float] = None,
        use_timing_context: bool = True,
        trigger_logging: bool = False,
        verbose_level: int = 1,
    ):
        if task_type.is_clf:
            if numerical_label:
                raise ValueError("numerical labels are invalid in CLASSIFICATION tasks")
        elif task_type.is_reg:
            if string_label:
                raise ValueError("string labels are invalid in REGRESSION tasks")
            if categorical_label:
                raise ValueError("categorical labels are invalid in REGRESSION tasks")
        self._simplify = simplify
        self._task_type = task_type
        self._time_series_config = time_series_config
        self.label_name = label_name
        self.string_label = string_label
        self.numerical_label = numerical_label
        self.categorical_label = categorical_label
        self._column_names = column_names
        self._valid_columns = valid_columns
        self._string_columns = string_columns
        self._numerical_columns = numerical_columns
        self._categorical_columns = categorical_columns
        self._process_methods = process_methods
        self._default_numerical_process = default_numerical_process
        self._default_categorical_process = default_categorical_process
        self._label_process_method = label_process_method
        self._numerical_threshold = numerical_threshold
        self._is_file = self._is_arr = False
        self._raw_dim: Optional[int] = None
        self._num_classes: Optional[int] = None
        self._label_idx: Optional[int]
        self._has_column_names: Optional[bool]
        self._delim: Optional[str]
        self._quote_char: Optional[str]
        self._raw: Optional[DataTuple] = None
        self._converted: Optional[DataTuple] = None
        self._processed: Optional[DataTuple] = None
        self._recognizers: Dict[int, Optional[Recognizer]]
        self._converters: Dict[int, Optional[Converter]]
        self._processors: Dict[int, Optional[Processor]]
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
        if time_idx is None:
            if time_name is None:
                msg = "either `time_column_name` or `time_column` should be provided"
                raise ValueError(msg)
            for k, v in self.column_names.items():
                if v == time_name:
                    time_idx = k
                    break
        raw = self.raw
        if raw is None:
            raise ValueError("`raw` should be provided in `ts_config`")
        raw_xt = raw.xT
        if raw_xt is None:
            raise ValueError("`raw.xT` should be provided in `ts_config`")
        assert isinstance(id_idx, int)
        assert isinstance(time_idx, int)
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
        return {"_recognizers", "_converters", "_processors"}

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
    def recognizers(self) -> Dict[int, Optional[Recognizer]]:
        return self._recognizers

    @property
    def converters(self) -> Dict[int, Optional[Converter]]:
        return self._converters

    @property
    def processors(self) -> Dict[int, Optional[Processor]]:
        return self._processors

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
        if self._recognizers[-1] is None:
            return TaskTypes.NONE
        self._task_type = TaskTypes.from_column_type(
            self._recognizers[-1].info.column_type,
            is_time_series=self.is_ts,
        )
        return self._task_type

    @property
    def column_names(self) -> Dict[int, str]:
        if self._column_names is None:
            self._column_names = {}
            for i in range(self.raw_dim):
                self._column_names.setdefault(i, str(i))
        return self._column_names

    def _get_prior_dict(
        self,
        attr: str,
        ts_value: Optional[bool],
    ) -> Dict[int, Optional[bool]]:
        prior_dict_attr = f"{attr}_dict"
        prior_dict = getattr(self, prior_dict_attr, None)
        if prior_dict is None:
            prior_columns = getattr(self, attr, None)
            prior_dict = {
                i: None if prior_columns is None else i in prior_columns
                for i in range(self.raw_dim)
            }
            setattr(self, prior_dict_attr, prior_dict)
        ts_config = self.ts_config
        if ts_config is not None:
            prior_dict[ts_config.id_column_idx] = ts_value
            prior_dict[ts_config.time_column_idx] = ts_value
        return prior_dict

    @property
    def prior_valid_columns(self) -> Dict[int, Optional[bool]]:
        return self._get_prior_dict("_valid_columns", True)

    @property
    def prior_string_columns(self) -> Dict[int, Optional[bool]]:
        return self._get_prior_dict("_string_columns", None)

    @property
    def prior_numerical_columns(self) -> Dict[int, Optional[bool]]:
        return self._get_prior_dict("_numerical_columns", False)

    @property
    def prior_categorical_columns(self) -> Dict[int, Optional[bool]]:
        return self._get_prior_dict("_categorical_columns", None)

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

    def _core_fit(self) -> "TabularData":
        if self._raw is None:
            raise ValueError("`_raw` is not provided")
        if self._raw.x is None:
            raise ValueError("`_raw.x` is not provided")
        self._raw_dim = len(self._raw.x[0])
        if self._simplify:
            self._recognizers = {}
            self._converters = {}
            self._processors = {}
            self._converted = self._processed = self._raw
        else:
            ts_indices = self.ts_indices
            # convert features
            features = self._raw.xT
            assert features is not None
            converted_features = []
            self._recognizers, self._converters = {}, {}
            for i, flat_arr in enumerate(features):
                column_name = self.column_names[i]
                is_valid = self.prior_valid_columns[i]
                is_string = self.prior_string_columns[i]
                is_numerical = self.prior_numerical_columns[i]
                is_categorical = self.prior_categorical_columns[i]
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
                kwargs: Dict[str, Any] = {
                    "is_valid": is_valid,
                    "is_string": is_string,
                    "is_numerical": is_numerical,
                    "is_categorical": is_categorical,
                }
                if self._numerical_threshold is not None:
                    kwargs["numerical_threshold"] = self._numerical_threshold
                with timing_context(self, "fit recognizer", enable=self._timing):
                    recognizer = Recognizer(column_name, **kwargs)  # type: ignore
                    recognizer.fit(flat_arr)
                    self._recognizers[i] = recognizer
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
                        self._converters[i] = converter
                    with timing_context(self, "convert", enable=self._timing):
                        converted_features.append(converter.converted_input)
            # convert labels
            if self._raw is None or self._raw.y is None:
                converted_labels = None
                self._recognizers[-1] = None
                self._converters[-1] = None
            else:
                with timing_context(self, "fit recognizer", enable=self._timing):
                    if self.label_name is None:
                        label_name = "__label__"
                    else:
                        label_name = self.label_name
                    recognizer = self._recognizers[-1] = Recognizer(
                        label_name,
                        is_label=True,
                        task_type=self._task_type,
                        is_valid=True,
                        is_string=self.string_label,
                        is_numerical=self.numerical_label,
                        is_categorical=self.categorical_label,
                        numerical_threshold=1.0,
                    )
                    recognizer.fit(self._flatten(self._raw.y))
                with timing_context(self, "fit converter", enable=self._timing):
                    converter = Converter.make_with(recognizer)
                    self._converters[-1] = converter
                with timing_context(self, "convert", enable=self._timing):
                    converted_labels = converter.converted_input.reshape([-1, 1])
            converted_x = np.vstack(converted_features).T
            # process features
            self._processors = {}
            processed_features = []
            previous_processors: List[Processor] = []
            idx = 0
            while idx < self.raw_dim:
                if idx in self.excludes or idx in ts_indices:
                    idx += 1
                    continue
                local_converter = self._converters[idx]
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
                self._processors[idx] = processor
                columns = converted_x[..., processor.input_indices]
                with timing_context(self, "fit processor", enable=self._timing):
                    processor.fit(columns)
                with timing_context(self, "process", enable=self._timing):
                    processed_features.append(processor.process(columns))
                idx += processor.input_dim
            # process labels
            if converted_labels is None:
                processed_labels = self._processors[-1] = None
            else:
                label_converter = self._converters[-1]
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
                    self._processors[-1] = processor.fit(converted_labels)
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
        self._valid_columns = [
            col for col in range(self.raw_dim) if col not in self.excludes
        ]
        self.ts_sorting_indices = None
        # time series
        if self.is_ts:
            self._get_ts_sorting_indices()
        # num classes
        if not self.is_reg and self._processed.y is not None:
            assert isinstance(self._processed.y, np.ndarray)
            self._num_classes = self._processed.y.max().item() + 1
        # prior settings
        _ = self.prior_valid_columns
        _ = self.prior_string_columns
        _ = self.prior_categorical_columns
        _ = self.prior_numerical_columns
        return self

    def _read_from_file(
        self,
        file_path: str,
        *,
        contains_labels: bool = True,
        label_idx: Optional[int] = None,
        has_column_names: Optional[bool] = None,
        quote_char: Optional[str] = None,
        delim: Optional[str] = None,
    ) -> "TabularData":
        self._is_file = True
        self._label_idx = label_idx
        self._has_column_names = has_column_names
        self._delim = delim
        self._quote_char = quote_char
        with timing_context(self, "read_file", enable=self._timing):
            x, y = self.read_file(file_path, contains_labels=contains_labels)
        self._raw = DataTuple.with_transpose(x, y)
        return self._core_fit()

    @staticmethod
    def _check_2d_y(y: data_type) -> None:
        failed = False
        if isinstance(y, list):
            failed = not isinstance(y[0], list)
        elif isinstance(y, np.ndarray):
            failed = len(y.shape) != 2
        if failed:
            raise ValueError("input labels should be 2d")

    def _read_from_arr(self, x: data_type, y: data_type) -> "TabularData":
        self._is_arr = True
        self._check_2d_y(y)
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
        label_converter = self._converters[-1]
        label_processor = self._processors[-1]
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
            return raw, raw
        # transform features
        features = raw.xT
        if features is None:
            raise ValueError("`raw` should contain `xT` for TabularData._transform")
        ts_indices = self.ts_indices
        converted_features_list = []
        for i, flat_arr in enumerate(features):
            if i in self.excludes or i in ts_indices:
                continue
            converter = self._converters[i]
            assert converter is not None
            converted_features_list.append(converter.convert(flat_arr))
        converted_features = np.vstack(converted_features_list)
        idx = 0
        processed = []
        while idx < self.raw_dim:
            if idx in self.excludes or idx in ts_indices:
                idx += 1
                continue
            processor = self._processors[idx]
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

    def _get_raw(
        self,
        x: Union[str, data_type],
        y: data_type = None,
        *,
        contains_labels: bool = True,
    ) -> DataTuple:
        if self._is_file:
            if isinstance(x, str):
                if y is not None:
                    raise ValueError(
                        "x refers to file_path but y is still provided "
                        f"({y}), which is illegal"
                    )
                x, y = self.read_file(x, contains_labels=contains_labels)
        return DataTuple.with_transpose(x, y)

    # API

    def read_file(
        self,
        file_path: str,
        *,
        contains_labels: bool = True,
    ) -> Tuple[str_data_type, Optional[str_data_type]]:
        ext = os.path.splitext(file_path)[1][1:]
        set_default = lambda n, default: n if n is not None else default
        if ext == "txt":
            has_column_names, delim, quote_char = map(
                set_default,
                [self._has_column_names, self._delim, self._quote_char],
                [False, " ", None],
            )
        elif ext == "csv":
            has_column_names, delim, quote_char = map(
                set_default,
                [self._has_column_names, self._delim, self._quote_char],
                [True, ",", '"'],
            )
        else:
            raise NotImplementedError(f"file type '{ext}' not recognized")
        self._delim, self._quote_char = delim, quote_char
        with open(file_path, "r") as f:
            first_row = None
            if has_column_names:
                first_row = column_names = f.readline().strip().split(delim)
                self._column_names = {i: name for i, name in enumerate(column_names)}
            data = []
            for line in f:
                elements = line.strip().split(delim)
                elements = ["nan" if not elem else elem for elem in elements]
                if quote_char is not None:
                    startswith_quote = [
                        elem.startswith(quote_char) for elem in elements
                    ]
                    endswith_quote = [elem.endswith(quote_char) for elem in elements]
                    merge_start, merge_intervals = None, []
                    for i, (startswith, endswith) in enumerate(
                        zip(startswith_quote, endswith_quote)
                    ):
                        if startswith and not endswith:
                            merge_start = i
                            continue
                        if endswith and not startswith and merge_start is not None:
                            merge_intervals.append((merge_start, i + 1))
                            merge_start = None
                            continue
                    idx, new_elements = 0, []
                    for start, end in merge_intervals:
                        if start > idx:
                            new_elements += elements[idx:start]
                        new_elements.append(delim.join(elements[start:end]))
                        idx = end
                    if idx < len(elements):
                        new_elements += elements[idx : len(elements)]
                    elements = new_elements
                if first_row is None:
                    first_row = elements
                else:
                    if len(first_row) != len(elements):
                        raise ValueError("num_features are not identical")
                data.append(elements)
        if not contains_labels:
            if self._raw is not None:
                if self._raw.x is None:
                    raise ValueError("`_raw.x` is not given")
                if len(data[0]) == len(self._raw.x[0]) + 1:
                    msg = "file contains labels but 'contains_labels=False' passed in"
                    raise ValueError(msg)
            return data, None

        label_idx: int
        if self._column_names is None or self.label_name is None:
            label_idx = -1 if self._label_idx is None else self._label_idx
        else:
            reverse_column_names: Dict[str, int]
            reverse_column_names = {v: k for k, v in self._column_names.items()}
            infer_label_idx = reverse_column_names.get(self.label_name)
            if infer_label_idx is None:
                raise ValueError(
                    f"'{self.label_name}' is not included in column names "
                    f"({list(self._column_names.values())})"
                )
            label_idx = infer_label_idx
        if label_idx < 0:
            label_idx = len(data[0]) + label_idx

        self._label_idx = label_idx
        x = [line[:label_idx] + line[label_idx + 1 :] for line in data]
        y = [line[label_idx : label_idx + 1] for line in data]
        return x, y

    def read(
        self,
        x: Union[str, data_type],
        y: Optional[Union[int, data_type]] = None,
        *,
        contains_labels: bool = True,
        **kwargs: Any,
    ) -> "TabularData":
        if isinstance(x, str):
            if y is not None and not isinstance(y, int):
                raise ValueError(
                    "`y` should be integer when `x` is a file. "
                    "In this case, `y` indicates the index of the label column."
                )
            self._read_from_file(
                x,
                label_idx=y,
                contains_labels=contains_labels,
                **kwargs,
            )
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
        raw = copied._raw = self._get_raw(x, y, contains_labels=contains_labels)
        converted, copied._processed = self._transform(raw)
        assert isinstance(converted.x, np.ndarray), "internal error occurred"
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
        copied._converters = copied_converters
        copied._converted = converted
        if copied.is_ts:
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
        raw = self._get_raw(x, y, contains_labels=contains_labels)
        bundle = self._transform(raw)
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
        label_processor = self._processors[-1]
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
                if data._simplify:
                    recognizers = {}
                    converters = {}
                    processors = {}
                else:
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
                data._recognizers = recognizers
                data._converters = converters
                data._processors = processors
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
                    converted_features = data._converted.x
                    converter_indices = [idx for idx in sorted(converters) if idx != -1]
                    for i, idx in enumerate(converter_indices):
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
        task_type: TaskTypes,
        *,
        simplify: bool = False,
        **kwargs: Any,
    ) -> "TabularData":
        if simplify:
            kwargs["simplify"] = simplify
        else:
            kwargs.setdefault("default_numerical_process", "identical")
            kwargs.setdefault("default_categorical_process", "identical")
        return cls(task_type=task_type, verbose_level=0, **kwargs)


__all__ = ["TabularData"]
