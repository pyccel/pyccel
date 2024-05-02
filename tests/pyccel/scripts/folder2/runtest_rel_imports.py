# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import pure
from .folder2_funcs import sum_to_n

@pure
def test_func():
    return sum_to_n(4)
