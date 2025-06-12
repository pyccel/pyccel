# pylint: disable=missing-function-docstring, missing-module-docstring

def array_int32_1d_add_const( x : 'Final[int32[:]]', y : 'int32[:]' ):
    x[:] += y
