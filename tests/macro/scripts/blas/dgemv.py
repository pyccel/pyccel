# pylint: disable=missing-function-docstring, missing-module-docstring/
# TODO - y must be optional
#      - t must be optiona, default 0 (map 1 -> 'T' and 0 -> 'N')
#      - default value for beta must be 0.0 and not 0

from pyccel.stdlib.internal.blas import dgemv
from numpy import zeros

n = 4
m = 5

a = zeros((n,m), 'double')
x = zeros(m, 'double')
y = zeros(n, 'double')

# ...
a[0,0] = 1.0
a[1,0] = 6.0
a[2,0] = 11.0
a[3,0] = 16.0

a[0,1] = 2.0
a[1,1] = 7.0
a[2,1] = 12.0
a[3,1] = 17.0

a[0,2] = 3.0
a[1,2] = 8.0
a[2,2] = 13.0
a[3,2] = 18.0

a[0,3] = 4.0
a[1,3] = 9.0
a[2,3] = 14.0
a[3,3] = 19.0

a[0,4] = 5.0
a[1,4] = 10.0
a[2,4] = 15.0
a[3,4] = 20.0
# ...

# ...
x[0] = 2.0
x[1] = 3.0
x[2] = 4.0
x[3] = 5.0
x[4] = 6.0
# ...

alpha = 2.0
beta  = 0.0

#$ header macro (y), _dgemv(alpha, a, x, y, t, beta=0, lda=a.shape[0], incx=1, incy=1) := dgemv(t, a.shape[0], a.shape[1], alpha, a, lda, x, incx, beta, y, incy)

y = _dgemv(alpha, a, x, y, 'N')
print(y)
