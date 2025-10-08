# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy import sin
from pyccel.decorators import inline

@inline
def sin_2(d : float):
    return sin(2*d)
