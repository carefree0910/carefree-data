cimport cython
cimport numpy as np

import unicodedata
import numpy as np
from libc.math cimport isnan


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
def transform_flat_data_with_dict(np.ndarray[np.float32_t, ndim=1] flat_data, transform_dict):
    cdef dict d = transform_dict
    cdef float elem
    cdef unsigned int i, n=len(flat_data)
    for i in range(n):
        elem = float(flat_data[i])
        if isnan(elem):
            flat_data[i] = d.get("nan", 0)
        else:
            flat_data[i] = d.get(elem, 0)
    return flat_data


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
def flat_arr_to_float32(arr x):
    cdef unsigned i, n=len(x)
    cdef np.ndarray[np.float32_t, ndim=1] rs=np.zeros(n, dtype=np.float32)
    for i in range(n):
        rs[i] = float(x[i])
    return rs
