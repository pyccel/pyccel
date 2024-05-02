# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.stdlib.internal.blas import dscal
from numpy import zeros

n   = 4

a = zeros(n, 'double')
b = zeros(n, 'double')

a[0] = 2.0
a[1] = 3.0
a[2] = 4.0
a[3] = 5.0

alpha = 2.0

#$ header macro (x), _dscal(alpha, x, n=x.shape, incx=1) := dscal(n, alpha, x, incx)
a = _dscal(alpha, a)
print(a)
