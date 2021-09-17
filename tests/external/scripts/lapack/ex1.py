# pylint: disable=missing-function-docstring, missing-module-docstring
# > Usage:
#
#   pyccel test.py -t
#   gfortran test.f90 -lblas -llapack
#   ./a.out

# TODO: - assert

from scipy.linalg.lapack import dgbtrf
from scipy.linalg.lapack import dgbtrs

from scipy.linalg.lapack import dgetrf
from scipy.linalg.lapack import dgetrs
#from scipy.linalg.lapack import dgecon

#from scipy.linalg.lapack import dgetri

from numpy import zeros, empty

def test_1():
    n   = 25
    ml  = 1
    mu  = 1
    lda = 2 * ml + mu + 1

    a = zeros((lda,n))
    b = zeros((1,n))

    b[0]   = 1.0
    b[n-1] = 1.0

    # Superdiagonal, Diagonal, Subdiagonal
    m = ml + mu
    a[m-1,1:n] = -1.0
    a[  m,0:n] =  2.0
    a[m+1,0:n-1] = -1.0

    a, ipiv, info = dgbtrf(a, ml, mu)
#    assert(info == 0)

    b, info = dgbtrs(a, ml, mu, b, ipiv)
#    assert(info == 0)

def test_2():
    n = 3
    lda = n

    a = zeros((lda,n))

    a[0,0] = 0.0
    a[0,1] = 1.0
    a[0,2] = 2.0

    a[1,0] = 4.0
    a[1,1] = 5.0
    a[1,2] = 6.0

    a[2,0] = 7.0
    a[2,1] = 8.0
    a[2,2] = 0.0

    a, ipiv, info = dgetrf(a)
#    assert(info == 0)

    #iwork = zeros(n, 'int')
    #lwork = 4 * n
    #work  = zeros(lwork)

    ## Get the condition number.
    #anorm = 1.0
    #rcond = -1.0
    # [TODO] dgecon('I', n, a, lda, anorm, rcond, work, iwork, info)
#    assert(info == 0)

def test_3():
    n = 3
    lda = n

    a = zeros((lda,n))

    a[0,0] = 0.0
    a[0,1] = 1.0
    a[0,2] = 2.0

    a[1,0] = 4.0
    a[1,1] = 5.0
    a[1,2] = 6.0

    a[2,0] = 7.0
    a[2,1] = 8.0
    a[2,2] = 0.0

    a, ipiv, info = dgetrf(a)
#    assert(info == 0)

    #lwork = 4 * n
    #work  = zeros(lwork)

    # Compute the inverse matrix.
    # [TODO] dgetri(n, a, lda, ipiv, work, lwork, info)
#    assert(info == 0)

def test_4():
    n = 3
    lda = n

    a = zeros((lda,n))

    a[0,0] = 0.0
    a[0,1] = 1.0
    a[0,2] = 2.0

    a[1,0] = 4.0
    a[1,1] = 5.0
    a[1,2] = 6.0

    a[2,0] = 7.0
    a[2,1] = 8.0
    a[2,2] = 0.0

    lu, ipiv, info = dgetrf(a)
#    assert(info == 0)

    # Compute the inverse matrix.
    b = zeros((1,n))
    b[0] = 14.0
    b[1] = 32.0
    b[2] = 23.0

    # Solve the linear system.
    x, info = dgetrs(a, ipiv, b, 'n')
#    assert(info == 0)

if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
    test_4()
