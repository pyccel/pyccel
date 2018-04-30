from pyccel.stdlib.internal.lapack import dgbtrf
from numpy import zeros

n   = 25
ml  = 1
mu  = 1
lda = 2 * ml + mu + 1

a = zeros((lda,n))

# Superdiagonal, Diagonal, Subdiagonal
m = ml + mu
a[m-1,1:n] = -1.0
a[  m,0:n] =  2.0
a[m+1,0:n-1] = -1.0

info = -1
ipiv = zeros(n, 'int')

#$ header macro (ab, ipiv, info), dgbtrf_v1(ab, kl, ku, m?, n?, ldab?) := dgbtrf(m|ab.shape[1], n|ab.shape[1], kl, ku, ab, ldab|ab.shape[0], ipiv, info)

a, ipiv, info = dgbtrf_v1(a, ml, mu)
