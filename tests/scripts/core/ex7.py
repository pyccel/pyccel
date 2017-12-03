#coding: utf-8

from numpy import zeros

a = zeros((10,10), double)

for i in range(0,10):
    a[i,i] = 2.0


for i in range(0,9):
    a[i,i+1] = -1.0

for i in range(0,9):
    a[i,i+1] = -1.0

n = 5
for i in range(0, n):
    x = 1
