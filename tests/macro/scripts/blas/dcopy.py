from pyccel.stdlib.internal.blas import dcopy
from numpy import zeros

n   = 4

a = zeros(n, 'double')
b = zeros(n, 'double')

a[0] = 2.0
a[1] = 3.0
a[2] = 4.0
a[3] = 5.0

#$ header macro (y), _dcopy(x, y, n?, incx?, incy?) := dcopy(n|x.shape, x, incx|1, y, incy|1)
b = _dcopy(a, b)
print(b)
