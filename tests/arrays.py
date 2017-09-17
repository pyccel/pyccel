# coding: utf-8

from numpy import zeros
from numpy import ones

n = int()
m = int()
n = 64
m = 5
a = ones(n, float)
b = ones(64, int)

f0 = 1.0
f1 = f0 + 2.0 * a[2]
f2 = a[2] + 2.0 * f1

a1 = zeros_like(a)
a2 = 2.0 * a + 1.0

c0 = zeros((m,n), double)
c1 = c0[0,1:3]
c2 = 2.0 * c0[0,1:3] + 1.0

d0 = zeros((m,n,n), double)
d1 = d0[0,1:3,0:4]
d2 = 2.0 * d0[0,1:3,0:4] + 1.0

r1 = dot(a, a)
r2 = 2.0 + 3.0 * dot(a, a)

i1 = dot(b, b)
i2 = 2 + 3 * dot(b, b)
