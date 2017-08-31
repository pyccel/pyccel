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

n1 = n-1
for k in range(0,n1):
    k1 = k+1
    for i in range(k1,n):
        t = a[i,k]
        if t <> 0.0:
            ak = a[k,k]
            lam = t/ak
            a[i,k1:n] = a[i,k1:n] - lam*a[k,k1:n]
            a[i,k] = lam
print(a)
