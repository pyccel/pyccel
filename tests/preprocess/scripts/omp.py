# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# this tests shows that OpenMP directives are moved to the appropriate parent

#$ omp for schedule(runtime)
for i in range(0, n):
    for j in range(0, m):
        a[i,j] = i-j
#$ omp end for nowait

def f(n,m,a):
    #$ omp for schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j
    #$ omp end for nowait

with toto:
    #$ omp for schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j
    #$ omp end for nowait


if init=='a':
    #$ omp for schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j
    #$ omp end for nowait
elif init=='b':
    #$ omp for schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            b[i,j] = i-j
    #$ omp end for nowait
else:
    #$ omp for schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            c[i,j] = i-j
    #$ omp end for nowait

