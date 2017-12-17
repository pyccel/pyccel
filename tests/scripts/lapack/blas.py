# coding: utf-8

#  pyccel blas.py --libdir=/home/macahr/projects/pyccel/third_party/lapack-3.7.1/usr/lib --libs="blas"
#  pyccel blas.py --libs="blas"

from pyccel.stdlib.external.blas import saxpy
from pyccel.stdlib.external.blas import daxpy

def test_blas_saxpy():
    n = 5
    sa = 1.0

    incx = 1
    sx = zeros(n, float)

    incy = 1
    sy = zeros(n, float)

    sx[0] = 1.0
    sx[1] = 3.0
    sx[3] = 5.0

    sy[0] = 2.0
    sy[1] = 4.0
    sy[3] = 6.0

    saxpy(n, sa, sx, incx, sy, incy)

def test_blas_daxpy():
    n = 5
    sa = 1.0

    incx = 1
    sx = zeros(n, double)

    incy = 1
    sy = zeros(n, double)

    sx[0] = 1.0
    sx[1] = 3.0
    sx[3] = 5.0

    sy[0] = 2.0
    sy[1] = 4.0
    sy[3] = 6.0

    daxpy(n, sa, sx, incx, sy, incy)

test_blas_saxpy()
test_blas_daxpy()
