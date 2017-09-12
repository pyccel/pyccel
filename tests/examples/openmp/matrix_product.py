# coding: utf-8

#from numpy import zeros

n = 500
m = 700
p = 500

a = zeros((n,m), double)
b = zeros((m,p), double)
c = zeros((n,p), double)

#$ omp parallel
#$ omp do schedule(runtime)
for i in range(0, n):
    for j in range(0, m):
        a[i,j] = i-j
#$ omp end do nowait

#$ omp do schedule(runtime)
for i in range(0, m):
    for j in range(0, p):
        b[i,j] = i+j
#$ omp end do nowait

#$ omp do schedule(runtime)
for i in range(0, n):
    for j in range(0, p):
        for k in range(0, p):
            c[i,j] = c[i,j] + a[i,k]*b[k,j]
#$ omp end do
#$ omp end parallel
