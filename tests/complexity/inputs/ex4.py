# coding: utf-8

n = int()
b = int()
m = int()
n = 10
b = 2
p = n / b

x = zeros((n,n), double)
y = zeros((n,n), double)
z = zeros((n,n), double)

r = zeros((b,b), double)
u = zeros((b,b), double)
v = zeros((b,b), double)

for i in range(0, p):
    for j in range(0, p):
        for k1 in range(0, b):
            for k2 in range(0, b):
                r[k1,k2] = z[i+k1,j+k2]
        for k in range(0, p):
            for k1 in range(0, b):
                for k2 in range(0, b):
                    u[k1,k2] = x[i+k1,k+k2]
                    v[k1,k2] = y[k+k1,j+k2]
            for ii in range(0, b):
                for jj in range(0, b):
                    for kk in range(0, b):
                        r[ii,jj] = r[ii,jj] + u[ii,kk]*v[kk,jj]
        for k1 in range(0, b):
            for k2 in range(0, b):
                z[i+k1,j+k2] = r[k1,k2]
