# coding: utf-8

n = 5
x = stencil(shape=3, step=2)
y = stencil(shape=(3,2), step=(2,2))
z = stencil(shape=(3,2,4), step=(2,2,2))

a = 2.0 * x[1, -1]
