# pylint: disable=missing-function-docstring, missing-module-docstring/
# Matrix inverse test using dgetri and dgetrf subroutines

from numpy import zeros,float64,int32
from pyccel.stdlib.internal.lapack import dgetrf
from pyccel.stdlib.internal.lapack  import dgetri

def dgetrf_test(A: 'float64[:,:](order=F)',piv:'int32[:]'):
    n = int32(len(A))
    info = int32(-1)
    lda = int32(n)

    print("Ready for lapack")
    dgetrf(n, n, A, lda, piv, info)
    print("Lapack ok")


def dgetri_test(A:'float64[:,:](order=F)'):
    n = int32(len(A))

    ipiv = zeros(n, dtype=int32,order='F')

    lda = int32(n)
    info = int32(-1)

    print("Calling dgetrf_test")
    dgetrf_test(A,ipiv)

    lwork = int32(4 * n)
    work  = zeros(lwork,dtype=float64)

    dgetri(n, A, lda, ipiv, work, lwork, info)


if __name__ == '__main__':
    print("Starting lapack_subroutine")
    N = 4

    A = zeros((N,N),dtype=float64,order='F')

    for i in range(N):
        A[i,i] = 1.0

    print("Calling dgetri_test")
    dgetri_test(A)

    print(A)
