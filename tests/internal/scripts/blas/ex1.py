# pylint: disable=missing-function-docstring, missing-module-docstring
# > Usage:
#
#   pyccel test.py -t
#   gfortran test.f90 -lblas
#   ./a.out

# TODO add saxpy test

from pyccel.stdlib.internal.blas import daxpy
import numpy as np

def test_daxpy():
    n = np.int32(5)
    sa = np.float64(1.0)

    incx = np.int32(1)
    sx = np.zeros(n)

    incy = np.int32(1)
    sy = np.zeros(n)

    sx[0] = np.float64(1.0)
    sx[1] = np.float64(3.0)
    sx[3] = np.float64(5.0)

    sy[0] = np.float64(2.0)
    sy[1] = np.float64(4.0)
    sy[3] = np.float64(6.0)

    daxpy(n, sa, sx, incx, sy, incy)

if __name__ == '__main__':
    test_daxpy()
