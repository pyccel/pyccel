# coding: utf-8

#  pyccel lapack.py --libdir=/home/macahr/projects/pyccel/third_party/lapack-3.7.1/usr/lib --libs="lapack"
#  pyccel lapack.py --libs="lapack"


from pyccel.stdlib.external.lapack import dgetrf
from pyccel.stdlib.external.lapack import dgetrs

def test_d():
    n = 3
    lda = n

    a = zeros((lda,n), double)

    a[0,0] = 0.0
    a[0,1] = 1.0
    a[0,2] = 2.0

    a[1,0] = 4.0
    a[1,1] = 5.0
    a[1,2] = 6.0

    a[2,0] = 7.0
    a[2,1] = 8.0
    a[2,2] = 0.0

    info = -1
    ipiv = zeros(n, int)

    dgetrf(n, n, a, lda, ipiv, info)
    assert(info == 0)

    # Compute the inverse matrix.
    b = zeros(n, double)
    b[0] = 14.0
    b[1] = 32.0
    b[2] = 23.0

    # Solve the linear system.
    dgetrs('n', n, 1, a, lda, ipiv, b, n, info)
    assert(info == 0)

test_d()
