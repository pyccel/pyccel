# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# this tests shows that OpenMP directives are moved to the appropriate parent

#$ omp do schedule(runtime)
for i in range(0, n):
    for j in range(0, m):
        a[i,j] = i-j
#$ omp end do nowait

def f(n,m,a):
    #$ omp do schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j
    #$ omp end do nowait

with toto:
    #$ omp do schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j
    #$ omp end do nowait


if init=='a':
    #$ omp do schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j
    #$ omp end do nowait
elif init=='b':
    #$ omp do schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            b[i,j] = i-j
    #$ omp end do nowait
else:
    #$ omp do schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            c[i,j] = i-j
    #$ omp end do nowait

