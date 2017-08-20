# coding: utf-8

from numpy import zeros

a = zeros(shape=64, dtype=float)
b = zeros(shape=8, dtype=int)

c = a[1]
d = c + 3 * a[2] + 1 - a[3]

e = a # not working. e must be declared as an array

x = zeros(shape=(2,8), dtype=float)
y = x[0,2]
