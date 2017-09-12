# coding: utf-8

from numpy import zeros

n = int()
m = int()
p = int()
n = 2

a = zeros((n,n), double)
b = zeros((n,n), double)

for i in range(0, n):
    for j in range(0, n):
        a[i,j] = i-j
        b[i,j] = i-j

for i in range(0, n):
    a[i,i] = n*n

print("> Initial matrix")
print(a)

for k in range(0,n-1):
    for i in range(k+1,n):
        t = a[i,k]
        if t <> 0.0:
            ak = a[k,k]
            lam = t/ak
            a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
            a[i,k] = lam

print("> Final matrix")
print(a)
