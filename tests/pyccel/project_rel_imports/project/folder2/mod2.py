# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types, pure
from ..folder1.mod1 import sum_to_n

@pure
@types('int')
def sum_to_n_squared(n):
    return sum_to_n(n)**2
