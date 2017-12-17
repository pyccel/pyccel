# coding: utf-8

#  pyccel lapack.py --libdir=/home/macahr/projects/pyccel/third_party/lapack-3.7.1/usr/lib --libs="lapack"
#  pyccel lapack.py --libs="lapack"

from pyccel.stdlib.external.lapack import dgbtrf
from pyccel.stdlib.external.lapack import dgbtrs

def test_1d():
    n   = 25
    ml  = 1
    mu  = 1
    lda = 2 * ml + mu + 1

    a = zeros((lda,n), double)
    b = zeros(n, double)

    b[0]   = 1.0
    b[n-1] = 1.0

    # Superdiagonal, Diagonal, Subdiagonal
    m = ml + mu
    a[m-1,1:n] = -1.0
    a[  m,0:n] =  2.0
    a[m+1,0:n-1] = -1.0

    info = -1
    ipiv = zeros(n, int)

    dgbtrf(n, n, ml, mu, a, lda, ipiv, info)
    assert(info == 0)

    dgbtrs('n', n, ml, mu, 1, a, lda, ipiv, b, n, info)
    assert(info == 0)

test_1d()
