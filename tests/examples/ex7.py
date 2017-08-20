# coding: utf-8

from numpy import zeros

a = zeros(shape=(10,10), dtype=float)

for i in range(0,10):
    for j in range(0,10):
        x = i - j

