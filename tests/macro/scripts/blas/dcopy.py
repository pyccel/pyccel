# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.stdlib.internal.blas import dcopy
from numpy import zeros

n   = 4

a = zeros(n, 'double')
b = zeros(n, 'double')

a[0] = 2.0
a[1] = 3.0
a[2] = 4.0
a[3] = 5.0

#$ header macro (y), _dcopy(x, y, n=x.shape, incx=1, incy=1) := dcopy(n, x, incx, y, incy)
b = _dcopy(a, b)
print(b)
