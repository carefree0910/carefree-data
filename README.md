# carefree-data

`carefree-data` implemented a data processing module with numpy.

#### Why carefree-data?

`carefree-data` is a data processing module which is capable of handling 'dirty' and 'messy' datasets.

##### For tabular datasets, `carefree-data` is able to:

+ Elegantly deal with data pre-processing.
    + A `Recognizer` to recognize whether a column is `STRING`, `NUMERICAL` or `CATEGORICAL`.
    + A `Converter` to convert a column into friendly format (["one", "two"] -> [0, 1]).
    + A `Processor` to further process columns (`OneHot`, `Normalize`, `MinMax`, ...).
    + And all the transforms could be inverse! (See `tests\unittests\test_tabular.py` -> `test_recover_labels` & `test_recover_features`).
    + And these procedures are all completed **AUTOMATICALLY**!
+ Handle datasets saved in files (`.txt`, `.csv`).
    + For `.txt`, `" "` will be the default `delimiter`.
    + For `.csv`, `","` will be the default `delimiter`, and the first row will be skipped as default.
    + `delimiter`, `label index`, `skip first` could be set manually.

#### Pandas-free

There is one more thing we'd like to mention: `carefree-data` is 'Pandas-free'. Pandas is an open source library providing easy-to-use data structures on structured datasets. Although it is a widely used library in almost every famous Machine Learning and Deep Learning module, we finally decided to escape from it, and the reasons are listed below:

+ `carefree-data` wants to have full control on the data, and Pandas is not flexible enough.
+ `carefree-data` needs higher performances. Pandas is fast, but not as fast as pure numpy (and sometimes cython) codes on some critical code paths.
+ Pandas provides many powerful functions, but `carefree-data` doesn't need that much, which means Pandas is a little 'heavy' for `carefree-data`.

In short, Pandas is a more general library, and that's why we've written some codes to cover our needs instead of directly utilizing it.


> Currently `carefree-data` only supports tabular datasets.


## Installation

`carefree-data` requires Python 3.6 or higher.

```bash
pip install carefree-data
```

or

```bash
git clone https://github.com/carefree0910/carefree-data.git
cd carefree-data
pip install -e .
```


## Basic Usages

### Get scikit-learn datasets

```python
from cfdata.tabular import TabularDataset

iris = TabularDataset.iris()
```

### Read from array / dataset

```python
from cfdata.tabular import *

iris = TabularDataset.iris()
x, y = iris.xy
assert TabularData().read(x, y) == TabularData.from_dataset(iris)
```

### Read from file

```python
from cfdata.tabular import TabularData

file = "/path/to/your/file"
data = TabularData().read(file)
assert data.processed == data.transform(file)
```


## License

`carefree-data` is MIT licensed, as found in the [`LICENSE`](https://github.com/carefree0910/carefree-data/blob/master/LICENSE) file.

---
