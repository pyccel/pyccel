# coding: utf-8

from pyccel.stdlib.parallel.openmp import Range
from pyccel.stdlib.parallel.openmp import Parallel

n = 50
m = 70
p = 50

a = zeros((n,m), double)
b = zeros((m,p), double)
c = zeros((n,p), double)

with Parallel(num_threads=2):
    for i in Range(0, n, 1, nowait=True):
        for j in range(0, m):
            a[i,j] = i-j

    for i in Range(0, m, 1, nowait=True):
        for j in range(0, p):
            b[i,j] = i+j

    for i in Range(0, n, 1, nowait=False):
        for j in range(0, p):
            for k in range(0, p):
                c[i,j] = c[i,j] + a[i,k]*b[k,j]
