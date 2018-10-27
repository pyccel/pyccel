from numpy import zeros
from numpy import ones
from numpy import sum
from numpy import array
from numpy import shape
from numpy import diag, cross

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

x1[:] = 2.

# TODO must give an error. Incompatible type
#b = int(a)

# TODO not working yet
#tt = zeros_like(x)

x17 = ones((n,m,2), 'double')
print(sum(x17)==n*m*2)

x18 = zeros((3, 3))
x19 = diag(x18)
x20 = diag(x18)
x21 = cross(x19, x20)

# TODO not working
#from numpy.random import random
#xr = random()
#print(xr)

# TODO not working
#from numpy.random import rand
#yr = rand()
#print(yr)
