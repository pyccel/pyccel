# coding: utf-8

from numpy import zeros

a = zeros(shape=64, dtype=float)
b = zeros(shape=8, dtype=int)

a[1] = 1
c = a[1]
d = c + 5.3 * a[1+1] + 4.0 - a[3]

e = a # not working. e must be declared as an array

x = zeros(shape=(2,8), dtype=float)
y = x[0,2]
