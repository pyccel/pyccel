# coding: utf-8

# This example is the python implementation of nowait.1.f from OpenMP 4.5 examples

n = 100
m = 100
n = int()
m = int()

a = zeros(shape=n, dtype=float)
b = zeros(shape=n, dtype=float)
y = zeros(shape=m, dtype=float)
z = zeros(shape=m, dtype=float)

#$ omp parallel
#$ omp do schedule(runtime)
for i in range(0, m):
    z[i] = i*1.0
#$ omp end do nowait

#$ omp do
for i in range(1, n):
    b[i] = a[i] / 2.0 + a[i-1] / 2.0
#$ omp end do nowait


#$ omp do
for i in range(1, m):
    y[i] = sqrt(z[i])
#$ omp end do nowait
#$ omp end parallel
