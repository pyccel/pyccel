# coding: utf-8

n = 500
m = 700
p = 500

a = zeros((n,m), double)
b = zeros((m,p), double)
c = zeros((n,p), double)

with parallel:
    for i in prange(0, n):
        for j in range(0, m):
            a[i,j] = i-j
