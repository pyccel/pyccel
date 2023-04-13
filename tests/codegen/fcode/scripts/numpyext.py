# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy import zeros
from numpy import ones
from numpy import sum as np_sum
from numpy import array
from numpy import shape
#from numpy import diag
from numpy import zeros_like
from numpy import full_like

x1 = array((1,2,3,5,8,5), 'float')
x2 = array((5,8,6,9,8,2), 'int')
x3 = array(((5,8,6,9,8,2),
           (5,8,6,9,8,2),
           (5,8,6,9,8,2),
           (5,8,6,9,8,2),
           (5,8,6,9,8,2),
           (5,8,6,9,8,2)),
          'double')

x4 = zeros((len(x1),len(x1)), 'int')

x1[:] = 2.

n = 5
x5 = zeros(4)
x6 = zeros(n)
x7 = zeros(n, 'int')

x8 = zeros((4, 3))
x9 = zeros((n, 2*n))

m = 5
x10 = ones(4)
x11 = ones(n)
x12 = ones(n, 'int')

x13 = ones((4, 3))
x14 = ones((n, 2*n))

x15 = array([1.,2.,3.])
x16 = x15

nn = shape(x14)
mm = shape(array([1.,2.,3.]))

x17 = ones((n,m,2), 'double')
print(np_sum(x17)==n*m*2)

x18 = zeros((3, 3))
#x19 = diag(x18)
#x20 = diag(x18)
#x21 = cross(x19, x20)

from numpy.random import random
xr = random()
print(xr)

from numpy.random import rand
yr = rand()
print(yr)

xa1 = zeros_like(x1)
xa2 = zeros_like(x2)
xa3 = zeros_like(x3)

xb1 = full_like(x1, 1)
xb2 = full_like(x2, 2)
xb3 = full_like(x3, 3)
