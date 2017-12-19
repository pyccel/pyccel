# coding: utf-8

#  pyccel lapack.py --libdir=/home/macahr/projects/pyccel/third_party/lapack-3.7.1/usr/lib --libs="lapack"
#  pyccel lapack.py --libs="lapack"

from pyccel.stdlib.external.lapack import dgbtrf
from pyccel.stdlib.external.lapack import dgbtrs

from pyccel.stdlib.external.lapack import dgetrf
from pyccel.stdlib.external.lapack import dgecon

from pyccel.stdlib.external.lapack import dgetrf
from pyccel.stdlib.external.lapack import dgetrs

from pyccel.stdlib.external.lapack import dgetrf
from pyccel.stdlib.external.lapack import dgetri

def test_1():
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

def test_2():
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

    # Get the condition number.
    anorm = 1.0
    rcond = -1.0
    dgecon('I', n, a, lda, anorm, rcond, work, iwork, info)
    assert(info == 0)

def test_3():
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

def test_4():
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

test_1()
test_2()
test_3()
test_4()
