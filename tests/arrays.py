# coding: utf-8

from numpy import zeros
from numpy import ones

a = ones(64, float)
b = ones(64, int)

f0 = 1.0
f1 = f0 + 2.0 * a[2]
f2 = a[2] + 2.0 * f1

#TODO allocate a1, a2
#a1 = a
#a2 = 2.0 * a1 + 1.0

r1 = dot(a, a)
r2 = 2.0 + 3.0 * dot(a, a)

i1 = dot(b, b)
i2 = 2 + 3 * dot(b, b)



