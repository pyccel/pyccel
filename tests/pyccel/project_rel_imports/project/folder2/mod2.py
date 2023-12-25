# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import pure
from ..folder1.mod1 import sum_to_n

@pure
def sum_to_n_squared(n : 'int'):
    return sum_to_n(n)**2
