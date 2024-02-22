# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.stdlib.internal.blas import dger
import numpy as np

if __name__ == '__main__':
    n = np.int32(4)
    m = np.int32(5)

    a = np.zeros((n,m), order='F')
    x = np.zeros(m)
    y = np.zeros(n)

    # ...
    x[0] = np.float64(2.0)
    x[1] = np.float64(3.0)
    x[2] = np.float64(4.0)
    x[3] = np.float64(5.0)
    x[4] = np.float64(6.0)
    # ...

    # ...
    y[0] = np.float64( 1.0)
    y[1] = np.float64(-1.0)
    y[2] = np.float64( 1.0)
    y[3] = np.float64(-1.0)
    # ...

    alpha = np.float64(2.0)
    incx = np.int32(1)
    incy = np.int32(1)

    dger(n, m, alpha, y, incy, x, incx, a, n)

    print(a)
