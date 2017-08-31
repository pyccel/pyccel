# coding: utf-8


def Ones(n):
    n = int()
    x = zeros(shape=n, dtype=float)
    for i in range(0, n):
        x[i] = 1
    return x

def Linspace(n):
    n = int()
    x = zeros(shape=n, dtype=float)
    k = n-1
    for i in range(0, n):
        x[i] = rational(i, k)
    return x
