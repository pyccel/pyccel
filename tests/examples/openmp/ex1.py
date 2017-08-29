# coding: utf-8

x = 0

#@ omp parallel private (x, y)
for i in range(0,10):
    x = x + 1
    y = 2*x
    print(y)
#@ omp end parallel
