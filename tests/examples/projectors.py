# coding: utf-8

#$ header legendre(int)
def legendre(p):
    k = p + 1
    x = zeros(k, double)
    w = zeros(k, double)
    if p == 1:
        x[0] = -0.577350269189625765
        x[1] =  0.577350269189625765
        w[0] =  1.0
        w[1] =  1.0
    elif p == 2:
        x[0] = -0.774596669241483377
        x[1] = 0.0
        x[2] = 0.774596669241483377
        w[0] = 0.55555555555555556
        w[1] = 0.888888888888888889
        w[2] = 0.55555555555555556
    elif p == 3:
        x[0] = -0.861136311594052575
        x[1] = -0.339981043584856265
        x[2] = 0.339981043584856265
        x[3] = 0.861136311594052575
        w[0] = 0.347854845137453853
        w[1] = 0.65214515486254615
        w[2] = 0.65214515486254614
        w[3] = 0.34785484513745386
    return x,w

#$ header knots_and_greville(int, int)
def knots_and_greville(n,p):
    n_elements = n-p
    m = n+p+1
    knots    = zeros(m, double)
    greville = zeros(n, double)
    for i in range(0, p+1):
        knots[i] = 0.0
    for i in range(p+1, n):
        j = i-p
        knots[i] = j / n_elements
    for i in range(n, n+p+1):
        knots[i] = 1.0
    for i in range(0, n):
        s = 0.0
        for j in range(i+1, i+p+1):
            s = s + knots[j]
        greville[i] = s / p
    return knots, greville

#$ header integrate_element_1d(double [:], double [:], double, double, int)
def integrate_element_1d(us, ws, x_min, x_max, p):
    r = 0.0
    d = x_max - x_min
    for j in range(0, p+1):
        u = us[j]
        w = ws[j]
        x = x_min + d * u
        w = 0.5 * d * w
        f = x * w
        r = r + f
    return r

#$ header integrate_1d(double [:], int, int)
def integrate_1d(t, n, p):
    n_elements = n-p
    us, ws = legendre(p)
    us = us + 1.0
    us = 0.5 * us
    rs = zeros(n_elements, double)
    for i in range(0,n-1):
        x_min = t[i]
        x_max = t[i+1]
    return rs

#        r = integrate_element_1d(us, ws, x_min, x_max, p)
#        rs[i] = r

n_elements = 4
p = 2
n = p+n_elements

knots, greville = knots_and_greville(n, p)
#r = integrate_1d(greville, n, p)
x_min = 0.0
x_max = 0.25
us, ws = legendre(p)
r = integrate_element_1d(us, ws, x_min, x_max, p)
print(r)
