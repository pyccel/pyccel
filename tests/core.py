# coding: utf-8

def matrix_product(a,b,n,m,p):
    a = array_2()
    b = array_2()
    n = int()
    m = int()
    p = int()
    k = int()
    c = zeros(shape=(n,p), dtype=float)
    for i in range(0, n):
        for j in range(0, p):
            for k in range(0, p):
                c[i,j] = c[i,j] + a[i,k]*b[k,j]
    return c

def Linspace(n):
    n = int()
    x = zeros(shape=n, dtype=float)
    l = n-1
    for i in range(0, n):
        x[i] = rational(i, l)
    return x
