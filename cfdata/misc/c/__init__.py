try:
    from .cython_wrappers import c_transform_data_with_dicts as transform_data_with_dicts
    from .cython_wrappers import c_get_counter_from_arr as get_counter_from_arr
    from .cython_wrappers import c_flat_str_to_float32 as flat_str_to_float32
    from .cython_wrappers import c_is_all_numeric as is_all_numeric
except ImportError:
    from .cython_substitute import naive_transform_data_with_dicts as transform_data_with_dicts
    from .cython_substitute import naive_get_counter_from_arr as get_counter_from_arr
    from .cython_substitute import naive_flat_str_to_float32 as flat_str_to_float32
    from .cython_substitute import naive_is_all_numeric as is_all_numeric


__all__ = [
    "transform_data_with_dicts", "get_counter_from_arr",
    "flat_str_to_float32", "is_all_numeric"
]
