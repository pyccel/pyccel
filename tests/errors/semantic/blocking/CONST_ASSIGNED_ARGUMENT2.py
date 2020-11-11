# pylint: disable=missing-function-docstring, missing-module-docstring/
#$ header function array_int32_1d_add_const(const int32[:], int32[:])
def array_int32_1d_add_const( x, y ):
    x[:] += y
