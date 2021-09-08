# pylint: disable=missing-function-docstring, missing-module-docstring, undefined-variable, unused-import
# TODO - beta, ta, tb must optional

from numpy import zeros
from pyccel.stdlib.internal.blas import dgemm

if __name__ == '__main__':
    m = 4
    k = 5
    n = 4

    a = zeros((m,k), 'double')
    b = zeros((k,n), 'double')
    c = zeros((m,n), 'double')

    # ...
    a[0,0] = 1.0
    a[1,0] = 6.0
    a[2,0] = 11.0
    a[3,0] = 16.0

    a[0,1] = 2.0
    a[1,1] = 7.0
    a[2,1] = 12.0
    a[3,1] = 17.0

    a[0,2] = 3.0
    a[1,2] = 8.0
    a[2,2] = 13.0
    a[3,2] = 18.0

    a[0,3] = 4.0
    a[1,3] = 9.0
    a[2,3] = 14.0
    a[3,3] = 19.0

    a[0,4] = 5.0
    a[1,4] = 10.0
    a[2,4] = 15.0
    a[3,4] = 20.0
    # ...

    # ...
    b[0,0] = 1.0
    b[1,2] = 1.0
    b[2,1] = 1.0
    b[3,3] = 1.0
    b[3,4] = 1.0
    # ...

    alpha = 2.0
    beta  = 1.0

    #$ header macro (c(:,:)), _dgemm(alpha, a, b, beta, c, ta, tb, lda=a.shape[0], ldb=b.shape[0], ldc=c.shape[0]) := dgemm(ta, tb, a.shape[0], b.shape[1], a.shape[1], alpha, a, lda, b, ldb, beta, c, ldc)

    c = _dgemm(alpha, a, b, beta, c, 'N', 'N')

    print(c)
