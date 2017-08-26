# coding: utf-8

def ones(n):
    n = int()
    x = zeros(shape=n, dtype=float)
    for i in range(0, n):
        x[i] = 1
    return x

def linspace(n):
    n = int()
    x = zeros(shape=n, dtype=float)
    k = n-1
    for i in range(0, n):
        x[i] = rational(i, k)
    return x

#m = int()
#m = 5
#y = linspace(m)
#print(y)


