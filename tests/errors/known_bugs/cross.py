# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy import ones, empty, cross
b = ones((2,3))
a = ones((2,3))
c = empty((2,3))

for i in range(2):
    a[i,:] = (1,2,3)
    b[i,:] = (5,0,4)

c = cross(a,b)

