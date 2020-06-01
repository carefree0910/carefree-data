cimport cython
cimport numpy as np

import unicodedata
import numpy as np
from libc.math cimport isnan
from collections import defaultdict


ctypedef fused arr:
    list
    tuple

# inner-core functions

def _is_numeric(s):
    try:
        s = float(s)
        return True
    except (TypeError, ValueError):
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            return False


# api

@cython.wraparound(False)
@cython.boundscheck(False)
def transform_data_with_dicts(np.ndarray[np.float32_t, ndim=2] data, list_of_idx, list_of_dict):
    cdef unsigned int td=len(list_of_idx)
    assert td == len(list_of_dict)
    if td == 0:
        return data
    cdef dict d
    cdef float elem
    cdef unsigned int i, j, idx, dim=len(data[0]), n=len(data)
    cdef np.ndarray[np.float32_t, ndim=1] sample
    cdef np.ndarray[np.float32_t, ndim=2] rs=np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        sample = data[i]
        for j in range(td):
            d = list_of_dict[j]
            idx = list_of_idx[j]
            elem = float(sample[idx])
            if isnan(elem):
                sample[idx] = d.get("nan", 0)
            else:
                sample[idx] = d.get(elem, 0)
        rs[i] = sample
    return rs


@cython.wraparound(False)
@cython.boundscheck(False)
def counter(arr x):
    cdef unsigned int i, n=len(x)
    d = defaultdict(int)
    for i in range(n):
        d[x[i]] += 1
    return d


@cython.wraparound(False)
@cython.boundscheck(False)
def is_all_numeric(arr x):
    cdef unsigned int i, n=len(x)
    for i in range(n):
        if not _is_numeric(x[i]):
            return False
    return True


@cython.wraparound(False)
@cython.boundscheck(False)
def flat_str_to_float32(arr x):
    cdef unsigned i, n=len(x)
    cdef np.ndarray[np.float32_t, ndim=1] rs=np.zeros(n, dtype=np.float32)
    for i in range(n):
        rs[i] = float(x[i])
    return rs
