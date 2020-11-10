# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types, pure
from project.folder1.mod1 import add2
from project.folder2.mod2 import sum_to_n_squared

@pure
@types('int')
def one_hundred_plus_sum_to_n_squared(n):
    return add2(100, sum_to_n_squared(n))
