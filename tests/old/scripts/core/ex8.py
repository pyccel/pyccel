# coding: utf-8


a = zeros(10, float)
for i in range(0,10):
    a[i] = i + 1
print(a)

b = zeros((4,4), float)
for i in range(0,4):
    for j in range(0,4):
        b[i,j] = i - j
print(b)

#vandermode matrix
v = zeros(10, float)
for i in range(0, 10):
    v[i] = i * 2

c = zeros((10,10), float)
for i in range(0, 10):
    for j in range(0, 10):
        c[i,j] = pow(v[i],10-j-i)
print(c)
