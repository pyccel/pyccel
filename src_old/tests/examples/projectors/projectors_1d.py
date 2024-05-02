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

#$ header make_knots(int, int)
def make_knots(n,p):
    n_elements = n-p
    m = n+p+1
    knots = zeros(m, double)
    for i in range(0, p+1):
        knots[i] = 0.0
    for i in range(p+1, n):
        j = i-p
        knots[i] = j / n_elements
    for i in range(n, n+p+1):
        knots[i] = 1.0
    return knots

#$ header make_greville(double [:], int, int)
def make_greville(knots, n, p):
    greville = zeros(n, double)
    for i in range(0, n):
        s = 0.0
        for j in range(i+1, i+p+1):
            s = s + knots[j]
        greville[i] = s / p
    return greville

#$ header func_V_0(double)
def func_V_0(x):
    y = 1.0-x
    y = x * y
    return y

#$ header func_V_1(double)
def func_V_1(x):
    y = 1.0-2.0*x
    return y

#$ header integrate_element_1d(double [:], double [:], double, double, int)
def integrate_element_1d(us, ws, x_min, x_max, p):
    r = 0.0
    d = x_max - x_min
    for j in range(0, p+1):
        u = us[j]
        w = ws[j]
        x = x_min + d * u
        w = 0.5 * d * w
        r = r + func_V_1(x) * w
    return r

#$ header interpolate_V_0(double [:], int, int)
def interpolate_V_0(t, n, p):
    n_elements = n-p
    rs = zeros(n_elements+1, double)
    for i in range(0, n_elements+1):
        rs[i] = func_V_0(t[i])
    return rs

#$ header interpolate_V_1(double [:], int, int)
def interpolate_V_1(t, n, p):
    n_elements = n-p
    us, ws = legendre(p)
    us = us + 1.0
    us = 0.5 * us
    rs = zeros(n_elements, double)
    for i in range(0, n_elements):
        x_min = t[i]
        x_max = t[i+1]
        r = integrate_element_1d(us, ws, x_min, x_max, p)
        rs[i] = r
    return rs


n_elements = 4
p = 2
n = p+n_elements

knots    = make_knots(n, p)
greville = make_greville(knots, n, p)
print("knots    = ", knots)
print("greville = ", greville)

r_0 = interpolate_V_0(greville, n, p)
r_1 = interpolate_V_1(greville, n, p)
print("r_0 = ", r_0)
print("r_1 = ", r_1)
