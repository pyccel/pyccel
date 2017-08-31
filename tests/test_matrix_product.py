# coding: utf-8

#from numpy import zeros
from core import matrix_product

n = int()
m = int()
p = int()
n = 2
m = 4
p = 2

a = zeros(shape=(n,m), dtype=float)
b = zeros(shape=(m,p), dtype=float)

for i in range(0, n):
    for j in range(0, m):
        a[i,j] = i-j

for i in range(0, m):
    for j in range(0, p):
        b[i,j] = i+j

c = matrix_product(a,b,n,m,p)
print(c)
