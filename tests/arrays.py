# coding: utf-8

from numpy import zeros
from numpy import ones

a = ones(64, float)

r0 = 1.0
r1 = r0 + 2.0 * a[2]
r2 = a[2] + 2.0 * r1

#TODO allocate a1, a2
a1 = a
a2 = 2.0 * a1 + 1.0



