# pylint: disable=missing-function-docstring, missing-module-docstring/
# TODO - y must be optional

from pyccel.stdlib.internal.blas import dger
from numpy import zeros

n = 4
m = 5

a = zeros((n,m), 'double')
x = zeros(m, 'double')
y = zeros(n, 'double')

# ...
x[0] = 2.0
x[1] = 3.0
x[2] = 4.0
x[3] = 5.0
x[4] = 6.0
# ...

# ...
y[0] =  1.0
y[1] = -1.0
y[2] =  1.0
y[3] = -1.0
# ...

alpha = 2.0
incx = 1
incy = 1

#$ header macro (a), _dger(alpha, x, y, a, incx=1, incy=1, lda=a.shape[0]) := dger(a.shape[0], a.shape[1], alpha, y, incy, x, incx, a, lda)

a = _dger(alpha, x, y, a)
print(a)
