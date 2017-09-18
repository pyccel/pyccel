# coding: utf-8

n = 10
m = 5
a0 = zeros((m,n), double)
a1 = a0[0,1:3]
a2 = 2.0 * a0[0,1:3] + 1.0

b0 = zeros((m,n,n), double)
b1 = b0[0,1:3,0:4]
b2 = 2.0 * b0[0,1:3,0:4] + 1.0

