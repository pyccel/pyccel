# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.stdlib.internal.blas import daxpy
from pyccel.stdlib.internal.blas import saxpy

#$header interface axpy=daxpy|saxpy

# > Usage:
#
#   pyccel test.py -t
#   gfortran test.f90 -lblas
#   ./a.out

# TODO add saxpy test
from numpy import zeros

def test_daxpy():
    n = 5
    sa = 1.0

    incx = 1
    sx = zeros(n)

    incy = 1
    sy = zeros(n)

    sx[0] = 1.0
    sx[1] = 3.0
    sx[3] = 5.0

    sy[0] = 2.0
    sy[1] = 4.0
    sy[3] = 6.0

    axpy(n, sa, sx, incx, sy, incy)

test_daxpy()
