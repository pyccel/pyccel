# coding: utf-8

#from numpy import zeros

a = zeros(shape=10, dtype=float)
for i0 in range(0,10):
    a[i0] = i0 + 1
print(a)

b = zeros(shape=(4,4), dtype=float)
for i1 in range(0,4):
    for j1 in range(0,4):
        b[i1,j1] = i1 - j1
print(b)

#vandermode matrix
v = zeros(shape=10, dtype=float)
for i2 in range(0, 10):
    v[i2] = i2 * 2

c = zeros(shape=(10,10), dtype=float)
for i3 in range(0, 10):
    for j3 in range(0, 10):
        k3 = 10-j3-1;
        c[i3,j3] = pow(v[i3],k3)
print(c)
