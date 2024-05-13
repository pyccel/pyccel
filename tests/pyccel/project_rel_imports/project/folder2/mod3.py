# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import pure
from ..folder1.mod1 import add2
from .mod2 import sum_to_n_squared

@pure
def one_hundred_plus_sum_to_n_squared(n : 'int'):
    return add2(100, sum_to_n_squared(n))
