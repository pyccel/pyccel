# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types('double[:,:]','double[:,:]')
def pairwise_python (X, D) :
    from numpy import sqrt, shape

    M, N = shape( X )
    for i in range (M) :
        for j in range (M) :
            r = 0.0
            for k in range (N) :
                tmp = X[ i , k ] - X[ j , k ]
                r += tmp * tmp
            D[ i , j ] = sqrt(r)

from numpy import zeros

s = 100
X = zeros([s, s])
rand = 0
a = 100
b = 821
m = 213
for i in range(s):
    for j in range(s):
        rand = (a * rand + b) % m
        X[i,j] = rand

D = zeros([s,s])
pairwise_python(X, D)

for i in range(s):
    for j in range(s):
        print(D[i,j])
