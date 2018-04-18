from numpy import zeros
from numpy import ones
from numpy import sum
from numpy import array
from numpy import shape

a = array((1,2,3,5,8,5), 'float')
b = array((5,8,6,9,8,2), 'int')
d = array(((5,8,6,9,8,2),
           (5,8,6,9,8,2),
           (5,8,6,9,8,2),
           (5,8,6,9,8,2),
           (5,8,6,9,8,2),
           (5,8,6,9,8,2)),
          'double')

k = zeros((len(a),len(a)), 'int')

n = 5
x1 = zeros(4)
x2 = zeros(n)
x3 = zeros(n, 'int')

y1 = zeros((4, 3))
y2 = zeros((n, 2*n))

m = 5
a1 = ones(4)
a2 = ones(n)
a3 = ones(n, 'int')

b1 = ones((4, 3))
b2 = ones((n, 2*n))

x = array([1.,2.,3.])
z = x

nn = shape(x)
mm = shape(array([1.,2.,3.]))

a = 2.
b = int(a)

# TODO not working yet
#tt = zeros_like(x)

z1 = ones((n,m,2), 'double')
print(sum(z1)==n*m*2)

# TODO not working
#from numpy.random import random
#xr = random()
#print(xr)

# TODO not working
#from numpy.random import rand
#yr = rand()
#print(yr)
