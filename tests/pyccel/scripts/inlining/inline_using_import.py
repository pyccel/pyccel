# pylint: disable=missing-function-docstring, missing-module-docstring
import my_func as f
from pyccel.decorators import inline

@inline
def func_2(d : float):
    return f.func(2*d)
