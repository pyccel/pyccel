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
    sa = 1.0

    incx = np.int32(1)
    sx = np.zeros(n)

    incy = np.int32(1)
    sy = np.zeros(n)

    sx[0] = 1.0
    sx[1] = 3.0
    sx[3] = 5.0

    sy[0] = 2.0
    sy[1] = 4.0
    sy[3] = 6.0

    daxpy(n, sa, sx, incx, sy, incy)

if __name__ == '__main__':
    test_daxpy()
