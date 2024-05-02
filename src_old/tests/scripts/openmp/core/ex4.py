# coding: utf-8

# This example is the python implementation of nowait.2.f from OpenMP 4.5 examples

n   = 100
np1 = n + 1

a = zeros(n, double)
b = zeros(n, double)
c = zeros(n, double)
y = zeros(np1, double)
z = zeros(n, double)

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
