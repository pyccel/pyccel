# coding: utf-8


n = 2
m = 4
p = 2

a = zeros((n,m), double)
b = zeros((m,p), double)
c = zeros((n,p), double)

for i in range(0, n):
    for j in range(0, m):
        a[i,j] = i-j

for i in range(0, m):
    for j in range(0, p):
        b[i,j] = i+j

for i in range(0, n):
    for j in range(0, p):
        for k in range(0, p):
            c[i,j] = c[i,j] + a[i,k]*b[k,j]

print(c)
