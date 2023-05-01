# pylint: disable=missing-function-docstring, missing-module-docstring
# Matrix inverse test using dgetri and dgetrf subroutines

from numpy import zeros,float64,int32
from pyccel.stdlib.internal.lapack import dgetrf
from pyccel.stdlib.internal.lapack  import dgetri

def dgetrf_test(A: 'float64[:,:](order=F)',piv:'int32[:]'):
    n = int32(len(A))
    info = int32(-1)
    lda = int32(n)

    dgetrf(n, n, A, lda, piv, info)


def dgetri_test(A:'float64[:,:](order=F)'):
    n = int32(len(A))

    ipiv = zeros(n, dtype=int32,order='F')

    lda = int32(n)
    info = int32(-1)

    dgetrf_test(A,ipiv)

    lwork = int32(4 * n)
    work  = zeros(lwork,dtype=float64)

    dgetri(n, A, lda, ipiv, work, lwork, info)


if __name__ == '__main__':
    N = 4

    A = zeros((N,N),dtype=float64,order='F')

    for i in range(N):
        A[i,i] = 1.0

    dgetri_test(A)

    print(A)
