# coding: utf-8

# This example is the python implementation of ploop.1.f from OpenMP 4.5 examples

n = 100
n = int()

a = zeros(shape=n, dtype=float)
b = zeros(shape=n, dtype=float)

#$ omp parallel
#$ omp do
for i in range(1, n):
    i1 = i-1
    i1 = int()
    b[i] = a[i] / 2.0 + a[i1] / 2.0
#$ omp end do
#$ omp end parallel
