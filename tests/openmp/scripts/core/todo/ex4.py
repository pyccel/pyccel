# coding: utf-8

# This example is the python implementation of nowait.2.f from OpenMP 4.5 examples

from numpy import zeros

n   = 100
np1 = n + 1

a = zeros(n)
b = zeros(n)
c = zeros(n)
y = zeros(np1)
z = zeros(n)

#$omp parallel
#$omp do schedule(static)
for i in range(0,n):
    c[i] = (a[i] + b[i]) / 2.0
#$omp end do nowait

#$omp do schedule(static)
for i in range(0,n):
    z[i] = sqrt(c[i])
#$omp end do nowait

#$omp do schedule(static)
for i in range(1,n+1):
    y[i] = z[i-1] + a[i]
#$omp end do nowait
#$omp end parallel
