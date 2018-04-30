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

#print('--- before swap')
#print(a)
#print(b)

#$ header macro dswap_v1(x, incx, y, incy) := dswap(x.shape, x, incx, y, incy)
dswap_v1(a, 1, b, 1)

#$ header macro dswap_v2(x, incx, y, incy?) := dswap(x.shape, x, incx, y, incy|1)
dswap_v2(a, 1, b)

#$ header macro dswap_v3(x, y, incx?, incy?) := dswap(x.shape, x, incx|1, y, incy|1)
dswap_v3(a, b)

#$ header macro (x, y), dswap_v4(x, y, incx?, incy?) := dswap(x.shape, x, incx|1, y, incy|1)
a,b = dswap_v4(a, b)

#print('--- after swap')
#print(a)
#print(b)
