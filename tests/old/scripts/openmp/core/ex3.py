# coding: utf-8

# This example is the python implementation of nowait.1.f from OpenMP 4.5 examples

n = 100
m = 100

a = zeros(n, double)
b = zeros(n, double)
y = zeros(m, double)
z = zeros(m, double)

#$ omp parallel
#$ omp do schedule(runtime)
for i in range(0, m):
    z[i] = i*1.0
#$ omp end do nowait

#$ omp do
for i in range(1, n):
    b[i] = (a[i] + a[i-1]) / 2.0
#$ omp end do nowait

#$ omp do
for i in range(1, m):
    y[i] = sqrt(z[i])
#$ omp end do nowait
#$ omp end parallel
