# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.stdlib.internal.blas import dswap
from numpy import zeros

n   = 4

a = zeros(n, 'double')
b = zeros(n, 'double')

a[0] = 2.0
a[1] = 3.0
a[2] = 4.0
a[3] = 5.0

b[0] = 5.0
b[1] = 4.0
b[2] = 9.0
b[3] = 2.0

print('--- before swap')
print(a)
print(b)

#$ header macro (x, y), _dswap(x, y, n=x.shape, incx=1, incy=1) := dswap(n, x, incx, y, incy)
a,b = _dswap(a, b)

print('--- after swap')
print(a)
print(b)
