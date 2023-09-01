# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy import array
from numpy import zeros_like

@elemental
def square(x : float):
    s = x*x
    return s

a = 2.0
b = square(a)
print(b)

xs = array([1., 2., 3.])
ys = zeros_like(xs)
ys = square(xs)
print(ys)
