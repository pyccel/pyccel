# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types, elemental
from numpy import array
from numpy import zeros_like

@elemental
@types(float)
def square(x):
    s = x*x
    return s

if __name__ == '__main__':
    a = 2.0
    b = square(a)
    print(b)

    xs = array([1., 2., 3.])
    ys = square(xs)
    print(ys)
