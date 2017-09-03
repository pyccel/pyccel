# coding: utf-8

#from numpy import zeros

n = int()
m = int()
p = int()
n = 2

a = zeros(shape=(n,n), dtype=float)
b = zeros(shape=(n,n), dtype=float)

for i in range(0, n):
    for j in range(0, n):
        a[i,j] = i-j
        b[i,j] = i-j
print(a)

#            a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
#            a[i,k] = lam

for k in range(0,n-1):
    for i in range(k+1,n):
        t = a[i,k]
        if t <> 0.0:
            ak = a[k,k]
            lam = t/ak
print(a)
