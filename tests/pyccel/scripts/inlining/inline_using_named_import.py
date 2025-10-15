# pylint: disable=missing-function-docstring, missing-module-docstring
from my_func import func
from pyccel.decorators import inline

@inline
def func_2(d : float):
    return func(2*d)
