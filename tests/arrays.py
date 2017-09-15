# coding: utf-8

from numpy import zeros
from numpy import ones

n = int()
n = 64
a = ones(n, float)
b = ones(64, int)

f0 = 1.0
f1 = f0 + 2.0 * a[2]
f2 = a[2] + 2.0 * f1

a1 = zeros_like(a)
a2 = 2.0 * a + 1.0

#r1 = dot(a, a)
#r2 = 2.0 + 3.0 * dot(a, a)
#
#i1 = dot(b, b)
#i2 = 2 + 3 * dot(b, b)



