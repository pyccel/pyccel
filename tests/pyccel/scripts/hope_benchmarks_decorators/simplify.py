# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types('double')
def y(x):
    from numpy import cos, sin
    return sin(x)**2 + (x**3 + x**2 - x - 1)/(x**2 + 2*x + 1) + cos(x)**2

print(y(2.78))
print(y(0.07))
