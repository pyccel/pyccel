# pylint: disable=missing-function-docstring, missing-module-docstring/
# TODO default value for alpha must double and not int
#      right now, textx raises an error, when we pass 1.0
from pyccel.stdlib.internal.blas import daxpy
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

alpha = 2.0

#$ header macro (y), _daxpy(x, y, alpha=1, n=x.shape, incx=1, incy=1) := daxpy(n, alpha, x, incx, y, incy)
b = _daxpy(a, b, alpha)
print(b)
