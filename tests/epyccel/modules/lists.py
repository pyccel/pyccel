# pylint: disable=missing-function-docstring, missing-module-docstring

__all__ = [
        'homogeneous_list_int',
        'homogeneous_list_bool',
        'homogeneous_list_float',
        'homogeneous_list_int_tuple'
]

def homogeneous_list_int():
    return list([1, 2, 3, 4])

def homogeneous_list_bool():
    return list([True, False, True, False])

def homogeneous_list_float():
    return list([1.0, 2.0, 3.0, 4.0])

def homogeneous_list_int_tuple():
    return list((1, 2, 3, 4))

