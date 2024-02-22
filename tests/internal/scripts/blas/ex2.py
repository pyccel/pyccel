# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.stdlib.internal.blas import dgemv
import numpy as np

if __name__ == '__main__':
    n = np.int32(4)
    m = np.int32(5)

    a = np.zeros((n,m), order='F')
    x = np.zeros(m)
    y = np.zeros(n)

    # ...
    a[0,0] = np.float64(1.0)
    a[1,0] = np.float64(6.0)
    a[2,0] = np.float64(11.0)
    a[3,0] = np.float64(16.0)

    a[0,1] = np.float64(2.0)
    a[1,1] = np.float64(7.0)
    a[2,1] = np.float64(12.0)
    a[3,1] = np.float64(17.0)

    a[0,2] = np.float64(3.0)
    a[1,2] = np.float64(8.0)
    a[2,2] = np.float64(13.0)
    a[3,2] = np.float64(18.0)

    a[0,3] = np.float64(4.0)
    a[1,3] = np.float64(9.0)
    a[2,3] = np.float64(14.0)
    a[3,3] = np.float64(19.0)

    a[0,4] = np.float64(5.0)
    a[1,4] = np.float64(10.0)
    a[2,4] = np.float64(15.0)
    a[3,4] = np.float64(20.0)
    # ...

    # ...
    x[0] = np.float64(2.0)
    x[1] = np.float64(3.0)
    x[2] = np.float64(4.0)
    x[3] = np.float64(5.0)
    x[4] = np.float64(6.0)
    # ...

    alpha = np.float64(2.0)
    beta  = np.float64(0.0)

    incx = np.int32(1)
    incy = np.int32(1)
    dgemv('N', n, m, alpha, a, n, x, incx, beta, y, incy)
