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

#$ header integrate_V_1(double [:], double [:], int, int, int, int)
def integrate_V_1(t_u, t_v, n_u, n_v, p_u, p_v):
    n_elements_u = n_u-p_u
    n_elements_v = n_v-p_v
    us, wus = legendre(p_u)
    vs, wvs = legendre(p_v)
    us = us + 1.0
    us = 0.5 * us
    vs = vs + 1.0
    vs = 0.5 * vs
    rs = zeros((n_elements_u, n_elements_v), double)
    for i_u in range(0, n_elements_u):
        u_min = t_u[i_u]
        u_max = t_u[i_u+1]
        for i_v in range(0, n_elements_v):
            v_min = t_v[i_v]
            v_max = t_v[i_v+1]
#            r = integrate_element_1d(us, ws, x_min, x_max, p)
#            rs[i] = r
    return rs

n_elements_u = 4
n_elements_v = 4
p_u = 2
p_v = 2
n_u = p_u + n_elements_u
n_v = p_v + n_elements_v

knots_u    = make_knots(n_u, p_u)
knots_v    = make_knots(n_v, p_v)
greville_u = make_greville(knots_u, n_u, p_u)
greville_v = make_greville(knots_v, n_v, p_v)
print("knots_u = ", knots_u)
print("knots_v = ", knots_v)
print("greville_u = ", greville_u)
print("greville_v = ", greville_v)

#r = integrate_1d(greville, n, p)
#print(r)
