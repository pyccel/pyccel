# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

# this tests shows that OpenMP directives are moved to the appropriate parent

#$ omp for nowait schedule(runtime)
for i in range(0, n):
    for j in range(0, m):
        a[i,j] = i-j

def f(n,m,a):
    #$ omp for nowait schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j

with toto:
    #$ omp for nowait schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j


if init=='a':
    #$ omp for nowait schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j
elif init=='b':
    #$ omp for nowait schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            b[i,j] = i-j
else:
    #$ omp for nowait schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            c[i,j] = i-j

