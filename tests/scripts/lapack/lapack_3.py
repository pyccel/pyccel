# coding: utf-8

#  pyccel lapack.py --libdir=/home/macahr/projects/pyccel/third_party/lapack-3.7.1/usr/lib --libs="lapack"
#  pyccel lapack.py --libs="lapack"


from pyccel.stdlib.external.lapack import dgetrf
from pyccel.stdlib.external.lapack import dgetri

def test_c():
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

    iwork = zeros(n, int)
    lwork = 4 * n
    work  = zeros(lwork, double)

    # Compute the inverse matrix.
    dgetri(n, a, lda, ipiv, work, lwork, info)
    assert(info == 0)

test_c()
