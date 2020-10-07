# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy import array
from numpy import empty
from numpy import ones

x = array([1, 2, 3])
y = empty((10, 10))

a = ones(3)
b = ones((4,3))
c = a+b
d = b+a
