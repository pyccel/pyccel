# coding: utf-8

x = 0

#@ omp parallel private (x, y, z)
for i in range(0,10):
    x = x + 1;
    y = 2*x
    for j in range(0,4):
        z = 2*y
#@ omp end parallel
