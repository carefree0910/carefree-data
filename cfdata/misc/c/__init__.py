try:
    from .cython_wrappers import (
        c_transform_flat_data_with_dict as transform_flat_data_with_dict,
    )
    from .cython_wrappers import c_flat_arr_to_float32 as flat_arr_to_float32
    from .cython_wrappers import c_is_all_numeric as is_all_numeric
except ImportError:
    from .cython_substitute import (
        naive_transform_flat_data_with_dict as transform_flat_data_with_dict,
    )
    from .cython_substitute import naive_flat_arr_to_float32 as flat_arr_to_float32
    from .cython_substitute import naive_is_all_numeric as is_all_numeric


__all__ = ["transform_flat_data_with_dict", "flat_arr_to_float32", "is_all_numeric"]
