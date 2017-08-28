# coding: utf-8

from core import dot

p = int()
p = 5
u = zeros(shape=p)
v = zeros(shape=p)
u[0] = 1
v[0] = 1
w = dot(u,v,p)

print(w)
