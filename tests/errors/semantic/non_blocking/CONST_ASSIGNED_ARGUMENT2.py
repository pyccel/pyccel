# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import Final #pylint: disable=unused-import

def array_int32_1d_add_const( x : 'Final[int32[:]]', y : 'int32[:]' ):
    x[:] += y
