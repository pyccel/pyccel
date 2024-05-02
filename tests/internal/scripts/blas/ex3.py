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
    incx = np.int32(1)
    incy = np.int32(1)

    dger(n, m, alpha, y, incy, x, incx, a, n)

    print(a)
