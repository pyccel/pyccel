# coding: utf-8

n = int()
n = 2

x = zeros((n,n), double)
y = zeros((n,n), double)
z = zeros((n,n), double)

for i in range(0, n):
    for j in range(0, n):
        for k in range(0, n):
            z[i,j] = z[i,j] + x[i,k]*y[k,j]
