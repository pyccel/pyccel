# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy import array, zeros_like
from pyccel.decorators import elemental


@elemental
def square(x: float):
    s = x * x
    return s


a = 2.0
b = square(a)
print(b)

xs = array([1.0, 2.0, 3.0])
ys = zeros_like(xs)
ys = square(xs)
print(ys)
