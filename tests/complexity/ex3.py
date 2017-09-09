# coding: utf-8

n = int()
n = 2

x = zeros(shape=(n,n), dtype=float)
y = zeros(shape=(n,n), dtype=float)
z = zeros(shape=(n,n), dtype=float)

for i in range(0, n):
    for j in range(0, n):
        for k in range(0, n):
            z[i,j] = z[i,j] + x[i,k]*y[k,j]
