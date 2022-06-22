# pylint: disable=missing-function-docstring, missing-module-docstring, no-value-for-parameter
from numpy import array

def multi_returns():
    x = array([1,2,3,4])
    z = array([1,2,3,4])
    return x, z
