# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from typing import Final
from pyccel.decorators import pure

@pure
def calculate_sum(arr : 'Final[float[:]]'):
    res = 0.0
    for a in arr:
        res += a
    return res
