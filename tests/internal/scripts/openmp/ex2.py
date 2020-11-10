# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# This example is the python implementation of ploop.1.f from OpenMP 4.5 examples

from numpy import zeros

n = 100

a = zeros(n)
b = zeros(n)

#$ omp parallel
#$ omp do
for i in range(1, n):
    b[i] = (a[i] + a[i-1]) / 2.0
#$ omp end do
#$ omp end parallel
