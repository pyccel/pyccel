# coding: utf-8

# This example is the python implementation of nowait.2.f from OpenMP 4.5 examples

n = int()
n = 100

n1 = int()
n1 = n+1

a = zeros(n, double)
b = zeros(n, double)
c = zeros(n, double)
y = zeros(n1, double)
z = zeros(n, double)

#$omp parallel
#$omp do schedule(static)
for i in range(0,n):
    s = a(i) + b(i)
    c[i] = s / 2.0
#$omp end do nowait

#$omp do schedule(static)
for i in range(0,n):
    t = c[i]
    z[i] = sqrt(t)
#$omp end do nowait

#$omp do schedule(static)
for i in range(1,n1):
    y[i] = z[i-1] + a[i]
#$omp end do nowait
#$omp end parallel
