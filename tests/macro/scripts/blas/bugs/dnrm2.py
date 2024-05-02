# pylint: disable=missing-function-docstring, missing-module-docstring/
# TODO not working yet
from pyccel.stdlib.internal.blas import dnrm2
from numpy import zeros

n   = 4

a = zeros(n, 'double')
b = zeros(n, 'double')

a[0] = 2.0
a[1] = 3.0
a[2] = 4.0
a[3] = 5.0

#$ header macro _dnrm2(x, n?, incx?) := dnrm2(n|x.shape, x, incx|1)
norm = 0.0
#norm = _dnrm2(a)
