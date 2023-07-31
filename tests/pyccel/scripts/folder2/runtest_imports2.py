# pylint: disable=missing-function-docstring, missing-module-docstring
from ..folder1.folder1_funcs import sum_to_n

@pure
def test_func():
    return sum_to_n(4)
