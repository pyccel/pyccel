# coding: utf-8

from numpy import zeros

a = zeros(shape=64, dtype=float)
b = zeros(shape=8, dtype=int)

a[1] = 1.0
a[2] = 1.0
a[3] = 1.0

c = a[1]

d = c + 5.3 * a[1+1] + 4.0 - a[3]
print(d)

e = zeros(shape=(2,8), dtype=float)
e[1,1] = 1

f = e[0,2]
print(f)

n = int()
n = 2
m = int()
m = 3
x = zeros(shape=(n,m,2), dtype=float)

for i in range(0, n):
    for j in range(0, m):
        x[i,j,0] = i-j
        x[i,j,1] = i+j
print(x)

y = zeros(shape=(n), dtype=float)
y[:2] = x[:2,0,0] + 1
print(y)

