# pylint: disable=missing-function-docstring, missing-module-docstring, undefined-variable, unused-import
from numpy import zeros
from pyccel.stdlib.internal.lapack import dgbtrf

if __name__ == '__main__':
    n   = 25
    ml  = 1
    mu  = 1
    lda = 2 * ml + mu + 1

    a = zeros((lda,n), dtype = 'double',order = 'F')

    # Superdiagonal, Diagonal, Subdiagonal
    m = ml + mu
    a[m-1,1:n] = -1.0
    a[  m,0:n] =  2.0
    a[m+1,0:n-1] = -1.0

    #$ header macro (ab(:,:), ipiv(ab.shape[0]), info), dgbtrf_v1(ab, kl, ku, m=ab.shape[1], n=ab.shape[1], ldab=ab.shape[0]) := dgbtrf(m, n, kl, ku, ab, ldab, ipiv, info)

    a, ipiv, info = dgbtrf_v1(a, ml, ku=mu)
