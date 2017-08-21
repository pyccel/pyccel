# coding: utf-8

# ex2_4 p40 from  NUMERICAL METHODS IN ENGINEERING WITH Python

# TODO not working yet

from numarray import zeros,Float64,array,product, diagonal,matrixmultiply
from gaussElimin import *

def vandermode(v):
    n = len(v)
    a = zeros((n,n),type=Float64)
    for j in range(n):
        a[:,j] = v**(n-j-1)
    return a

v = array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
b = array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

a = vandermode(v)

aOrig = a.copy() # Save original matrix
bOrig = b.copy() # and the constant vector

x = gaussElimin(a,b)
det = product(diagonal(a))

print 'x =\n',x
print '\ndet =',det

print '\nCheck result: [a] { x } - b =\n', matrixmultiply(aOrig,x) - bOrig
