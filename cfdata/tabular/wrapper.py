import os
import logging

import numpy as np

from typing import *
from cftool.misc import SavingMixin, timing_context

from .types import *
from .recognizer import *
from .converters import *
from .processors import *


class TabularData(SavingMixin):
    def __init__(self,
                 *,
                 task_type: TaskTypes = None,
                 label_name: str = "label",
                 string_label: bool = False,
                 numerical_label: bool = False,
                 categorical_label: bool = False,
                 column_names: Dict[int, str] = None,
                 valid_columns: List[int] = None,
                 string_columns: List[int] = None,
                 numerical_columns: List[int] = None,
                 categorical_columns: List[int] = None,
                 process_methods: Dict[int, str] = None,
                 label_process_method: str = None,
                 numerical_threshold: float = None,
                 trigger_logging: bool = False,
                 verbose_level: int = 1):
        if task_type is not None:
            if task_type is TaskTypes.CLASSIFICATION:
                if numerical_label:
                    raise ValueError("numerical labels are invalid in CLASSIFICATION tasks")
            else:
                if string_label:
                    raise ValueError("string labels are invalid in REGRESSION tasks")
                if categorical_label:
                    raise ValueError("categorical labels are invalid in REGRESSION tasks")
        self._task_type = task_type
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
        self._label_process_method = label_process_method
        self._numerical_threshold = numerical_threshold
        self._is_file = self._is_arr = False
        self._label_idx = self._skip_first = self._delim = None
        self._raw = self._converted = self._processed = None
        self._recognizers = self._converters = self._processors = None
        self._init_logging(verbose_level, trigger_logging)
        self.excludes = set()

    def __eq__(self, other: "TabularData"):
        if self.raw != other.raw:
            return False
        if self.converted != other.converted:
            return False
        return self.processed == other.processed

    @property
    def cache_excludes(self):
        return {"_recognizers", "_converters", "_processors"}

    @property
    def data_tuple_base(self) -> Type[NamedTuple]:
        return DataTuple

    @property
    def data_tuple_attributes(self) -> List[str]:
        return ["_raw", "_converted", "_processed"]

    @property
    def raw(self) -> DataTuple:
        return self._raw

    @property
    def converted(self) -> DataTuple:
        return self._converted

    @property
    def processed(self) -> DataTuple:
        return self._processed

    @property
    def recognizers(self) -> Dict[int, Recognizer]:
        return self._recognizers

    @property
    def converters(self) -> Dict[int, Converter]:
        return self._converters

    @property
    def processors(self) -> Dict[int, Processor]:
        return self._processors

    @property
    def raw_dim(self) -> int:
        return len(self._raw.x[0])

    @property
    def processed_dim(self) -> int:
        return self._processed.x.shape[1]

    @property
    def task_type(self) -> Union[TaskTypes, None]:
        if self._task_type is not None:
            return self._task_type
        if self._recognizers[-1] is None:
            return
        return TaskTypes.from_column_type(self._recognizers[-1].info.column_type)

    @property
    def column_names(self) -> Dict[int, str]:
        if self._column_names is None:
            self._column_names = {}
        for i in range(self.raw_dim):
            self._column_names.setdefault(i, str(i))
        return self._column_names

    def _get_dict(self, attr: str) -> Dict[int, bool]:
        force_dict_attr = f"{attr}_dict"
        force_dict = getattr(self, force_dict_attr, None)
        if force_dict is None:
            force_columns = getattr(self, attr, None)
            if force_columns is None:
                force_columns = []
            force_dict = {i: i in force_columns for i in range(self.raw_dim)}
            setattr(self, force_dict_attr, force_dict)
        return force_dict

    @property
    def force_valid_columns(self) -> Dict[int, bool]:
        return self._get_dict("_valid_columns")

    @property
    def force_string_columns(self) -> Dict[int, bool]:
        return self._get_dict("_string_columns")

    @property
    def force_numerical_columns(self) -> Dict[int, bool]:
        return self._get_dict("_numerical_columns")

    @property
    def force_categorical_columns(self) -> Dict[int, bool]:
        return self._get_dict("_categorical_columns")

    # Core

    def _read_file(self,
                   file_path: str) -> Tuple[raw_data_type, raw_data_type]:
        ext = os.path.splitext(file_path)[1][1:]
        set_default = lambda n, default: n if n is not None else default
        if ext == "txt":
            skip_first, delim = map(set_default, [self._skip_first, self._delim], [False, " "])
        elif ext == "csv":
            skip_first, delim = map(set_default, [self._skip_first, self._delim], [True, ","])
        else:
            raise NotImplementedError(f"file type '{ext}' not recognized")
        with open(file_path, "r") as f:
            if skip_first:
                f.readline()
            data = [["nan" if not elem else elem for elem in line.strip().split(delim)] for line in f]
        if self._label_idx is None:
            if len(data[0]) != len(self._raw.x[0]):
                raise ValueError("file contains labels but 'contains_labels=False' passed in")
            return data, None
        if self._label_idx < 0:
            self._label_idx = len(data[0]) + self._label_idx
        x = [line[:self._label_idx] + line[self._label_idx+1:] for line in data]
        y = [line[self._label_idx:self._label_idx+1] for line in data]
        return x, y

    @staticmethod
    def _flatten(data: data_type) -> data_type:
        if isinstance(data, list):
            return sum(data, [])
        return data.ravel()

    def _core_fit(self) -> "TabularData":
        with timing_context(self, "convert"):
            # convert features
            features = self._raw.xT
            converted_features = []
            self._recognizers, self._converters = {}, {}
            for i, flat_arr in enumerate(features):
                column_name = self.column_names[i]
                force_valid = self.force_valid_columns[i]
                force_string = self.force_string_columns[i]
                force_numerical = self.force_numerical_columns[i]
                force_categorical = self.force_categorical_columns[i]
                if i == self.raw_dim - 1 == len(self.excludes):
                    if i > 0:
                        self.log_msg(
                            f"last column {column_name} is forced to be valid "
                            "because previous columns are all excluded", self.warning_prefix,
                            verbose_level=2, msg_level=logging.WARNING
                        )
                    force_valid = True
                kwargs = {
                    "force_string": force_string,
                    "force_numerical": force_numerical,
                    "force_categorical": force_categorical,
                    "force_valid": force_valid
                }
                if self._numerical_threshold is not None:
                    kwargs["numerical_threshold"] = self._numerical_threshold
                with timing_context(self, "fit recognizer"):
                    recognizer = self._recognizers[i] = Recognizer(column_name, **kwargs).fit(flat_arr)
                if not recognizer.info.is_valid:
                    self.log_msg(recognizer.info.msg, self.warning_prefix, 2, logging.WARNING)
                    self.excludes.add(i)
                    continue
                with timing_context(self, "fit converter"):
                    converter = self._converters[i] = Converter.make_with(recognizer)
                converted_features.append(converter.converted_input)
            # convert labels
            if self._raw.y is None:
                converted_labels = self._recognizers[-1] = self._converters[-1] = None
            else:
                with timing_context(self, "fit recognizer"):
                    recognizer = self._recognizers[-1] = Recognizer(
                        self.label_name,
                        is_label=True,
                        task_type=self._task_type,
                        force_valid=True,
                        force_string=self.string_label,
                        force_numerical=self.numerical_label,
                        force_categorical=self.categorical_label,
                        numerical_threshold=1.
                    ).fit(self._flatten(self._raw.y))
                with timing_context(self, "fit converter"):
                    converter = self._converters[-1] = Converter.make_with(recognizer)
                converted_labels = converter.converted_input.reshape([-1, 1])
        converted_x = np.vstack(converted_features).T
        with timing_context(self, "process"):
            # process features
            self._processors = {}
            processed_features = []
            previous_processors = []
            idx = 0
            while idx < self.raw_dim:
                if idx in self.excludes:
                    idx += 1
                    continue
                column_type = self._converters[idx].info.column_type
                method = None
                if self._process_methods is not None:
                    method = self._process_methods.get(idx)
                if method is None:
                    method = "normalize" if column_type is ColumnTypes.NUMERICAL else "one_hot"
                processor = self._processors[idx] = processor_dict[method](previous_processors)
                previous_processors.append(processor)
                columns = converted_x[..., processor.input_indices]
                with timing_context(self, "fit processor"):
                    processor.fit(columns)
                with timing_context(self, "process with processor"):
                    processed_features.append(processor.process(columns))
                idx += processor.input_dim
            # process labels
            if converted_labels is None:
                processed_labels = self._processors[-1] = None
            else:
                column_type = self._converters[-1].info.column_type
                method = None
                if self._label_process_method is not None:
                    method = self._label_process_method
                if method is None:
                    method = "normalize" if column_type is ColumnTypes.NUMERICAL else "identical"
                with timing_context(self, "fit processor"):
                    processor = self._processors[-1] = processor_dict[method]([]).fit(converted_labels)
                with timing_context(self, "process with processor"):
                    processed_labels = processor.process(converted_labels)
        if self.task_type is TaskTypes.CLASSIFICATION:
            converted_labels = converted_labels.astype(np.int)
            processed_labels = processed_labels.astype(np.int)
        self._converted = DataTuple(converted_x, converted_labels)
        self._processed = DataTuple(np.hstack(processed_features), processed_labels)
        return self

    def _read_from_file(self,
                        file_path: str,
                        *,
                        label_idx: int = -1,
                        skip_first: bool = None,
                        delim: str = None) -> "TabularData":
        self._is_file = True
        self._label_idx, self._skip_first, self._delim = label_idx, skip_first, delim
        with timing_context(self, "_read_file"):
            x, y = self._read_file(file_path)
        self._raw = DataTuple.with_transpose(x, y)
        return self._core_fit()

    def _read_from_arr(self,
                       x: data_type,
                       y: data_type) -> "TabularData":
        self._is_arr = True
        self._raw = DataTuple.with_transpose(x, y)
        return self._core_fit()

    def _transform(self,
                   raw: DataTuple,
                   return_converted: bool) -> Union[DataTuple, Tuple[DataTuple, DataTuple]]:
        # transform features
        features = raw.xT
        converted_features = np.vstack([
            self._converters[i].convert(flat_arr)
            for i, flat_arr in enumerate(features) if i not in self.excludes
        ])
        idx = 0
        processed = []
        while idx < self.raw_dim:
            if idx in self.excludes:
                idx += 1
                continue
            processor = self._processors[idx]
            input_indices = processor.input_indices
            columns = processor.process(converted_features[input_indices].T)
            processed.append(columns)
            idx += processor.input_dim
        transformed_features = np.hstack(processed)
        # transform labels
        if raw.y is None:
            converted_labels = transformed_labels = None
        else:
            converted_labels = self._converters[-1].convert(self._flatten(raw.y))
            transformed_labels = self._processors[-1].process(converted_labels.reshape([-1, 1]))
        # check categorical
        if self.task_type is TaskTypes.CLASSIFICATION:
            converted_labels = converted_labels.astype(np.int)
            transformed_labels = transformed_labels.astype(np.int)
        transformed = DataTuple(transformed_features, transformed_labels)
        if not return_converted:
            return transformed
        converted = DataTuple(converted_features, converted_labels)
        return converted, transformed

    # API

    def read(self,
             x: Union[str, data_type],
             y: Union[int, data_type] = -1,
             **kwargs) -> "TabularData":
        if isinstance(x, str):
            self._read_from_file(x, label_idx=y, **kwargs)
        else:
            if isinstance(y, int):
                y = None
            self._read_from_arr(x, y)
        self.log_timing()
        return self

    def transform(self,
                  x: Union[str, data_type],
                  y: data_type = None,
                  *,
                  return_converted: bool = False) -> Union[DataTuple, Tuple[DataTuple, DataTuple]]:
        if self._is_file:
            x, y = self._read_file(x)
        raw = DataTuple.with_transpose(x, y)
        return self._transform(raw, return_converted)

    def recover_labels(self,
                       y: np.ndarray,
                       *,
                       inplace: bool = False) -> np.ndarray:
        process_recovered = self._processors[-1].recover(y, inplace=inplace)
        convert_recovered = self.converters[-1].recover(process_recovered.ravel(), inplace=inplace)
        return convert_recovered.reshape([-1, 1])

    def load(self, folder, *, compress=True) -> "TabularData":
        super().load(folder)
        is_file, is_arr = self._is_file, self._is_arr
        self.read(*self._raw[:2])
        self._is_file, self._is_arr = is_file, is_arr
        return self

    def to_dataset(self) -> TabularDataset:
        return TabularDataset(*self.processed.xy, task_type=self.task_type)

    @classmethod
    def from_dataset(cls,
                     dataset: TabularDataset,
                     **kwargs):
        task_type = kwargs.pop("task_type", dataset.task_type)
        return cls(task_type=task_type, **kwargs).read(*dataset.xy)


__all__ = ["TabularData"]
