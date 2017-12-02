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

#$ header f_scalar(double, double)
def f_scalar(x, y):
    f = x * y
    return f

#$ header f_vector(double, double)
def f_vector(x, y):
    f1 = x*y
    f2 = x*y
    return f1, f2

#$ header integrate_edge(int, int, double, double [:], double [:], double, double, int)
def integrate_edge(component, axis, y, us, ws, x_min, x_max, p):
    r = 0.0
    d = x_max - x_min
    for j in range(0, p+1):
        u = us[j]
        w = ws[j]
        x = x_min + d * u
        w = 0.5 * d * w
        if axis==0:
            f1, f2 = f_vector(x, y)
        else:
            f1, f2 = f_vector(y, x)
        if component == 0:
            f = f1
        else:
            f = f2
        r = r + f * w
    return r

#$ header interpolate_V_0(double [:], double [:], int, int, int, int)
def interpolate_V_0(t_u, t_v, n_u, n_v, p_u, p_v):
    n_elements_u = n_u-p_u
    n_elements_v = n_v-p_v

    nu1 = n_elements_u+1
    nv1 = n_elements_v+1

    r = zeros(nu1*nv1, double)
    i = 0
    for i_u in range(0, n_elements_u+1):
        for i_v in range(0, n_elements_v+1):
            r[i] = f_scalar(t_u[i_u], t_v[i_v])
            i = i + 1
    return r

#$ header interpolate_V_1(double [:], double [:], int, int, int, int)
def interpolate_V_1(t_u, t_v, n_u, n_v, p_u, p_v):
    n_elements_u = n_u-p_u
    n_elements_v = n_v-p_v
    us, wus = legendre(p_u)
    vs, wvs = legendre(p_v)
    us = us + 1.0
    us = 0.5 * us
    vs = vs + 1.0
    vs = 0.5 * vs

    nu1 = n_elements_u
    nv1 = n_elements_v+1
    nu2 = n_elements_u+1
    nv2 = n_elements_v

    r_0 = zeros((nu1, nv1), double)
    r_1 = zeros((nu2, nv2), double)

    component = 0
    axis      = 0
    for i_u in range(0, n_elements_u):
        x_min = t_u[i_u]
        x_max = t_u[i_u+1]
        for i_v in range(0, n_elements_v+1):
            y = t_v[i_v]
            r_0[i_u, i_v] = integrate_edge(component, axis, y, us, wus, x_min, x_max, p_u)

    component = 1
    axis      = 1
    for i_u in range(0, n_elements_u+1):
        y = t_u[i_u]
        for i_v in range(0, n_elements_v):
            x_min = t_v[i_v]
            x_max = t_v[i_v+1]
            r_1[i_u, i_v] = integrate_edge(component, axis, y, vs, wvs, x_min, x_max, p_v)

    m = nu1 * nv1 + nu2 * nv2
    r = zeros(m, double)
    i = 0
    for i_u in range(0, nu1):
        for i_v in range(0, nv1):
            r[i] = r_0[i_u, i_v]
            i = i + 1
    for i_u in range(0, nu2):
        for i_v in range(0, nv2):
            r[i] = r_1[i_u, i_v]
            i = i + 1
    return r

#$ header interpolate_V_2(double [:], double [:], int, int, int, int)
def interpolate_V_2(t_u, t_v, n_u, n_v, p_u, p_v):
    n_elements_u = n_u-p_u
    n_elements_v = n_v-p_v
    us, wus = legendre(p_u)
    vs, wvs = legendre(p_v)
    us = us + 1.0
    us = 0.5 * us
    vs = vs + 1.0
    vs = 0.5 * vs

    nu1 = n_elements_u+1
    nv1 = n_elements_v
    nu2 = n_elements_u
    nv2 = n_elements_v+1

    r_0 = zeros((nu1, nv1), double)
    r_1 = zeros((nu2, nv2), double)

    component = 0
    axis      = 1
    for i_u in range(0, n_elements_u+1):
        y = t_u[i_u]
        for i_v in range(0, n_elements_v):
            x_min = t_v[i_v]
            x_max = t_v[i_v+1]
            r_0[i_u, i_v] = integrate_edge(component, axis, y, vs, wvs, x_min, x_max, p_v)

    component = 1
    axis      = 0
    for i_u in range(0, n_elements_u):
        x_min = t_u[i_u]
        x_max = t_u[i_u+1]
        for i_v in range(0, n_elements_v+1):
            y = t_v[i_v]
            r_1[i_u, i_v] = integrate_edge(component, axis, y, us, wus, x_min, x_max, p_u)

    m = nu1 * nv1 + nu2 * nv2
    r = zeros(m, double)
    i = 0
    for i_u in range(0, nu1):
        for i_v in range(0, nv1):
            r[i] = r_0[i_u, i_v]
            i = i + 1
    for i_u in range(0, nu2):
        for i_v in range(0, nv2):
            r[i] = r_1[i_u, i_v]
            i = i + 1
    return r

#$ header interpolate_V_3(double [:], double [:], int, int, int, int)
def interpolate_V_3(t_u, t_v, n_u, n_v, p_u, p_v):
    n_elements_u = n_u-p_u
    n_elements_v = n_v-p_v

    us, wus = legendre(p_u)
    vs, wvs = legendre(p_v)
    us = us + 1.0
    us = 0.5 * us
    vs = vs + 1.0
    vs = 0.5 * vs

    r = zeros(n_elements_u*n_elements_v, double)
    i = 0
    for i_u in range(0, n_elements_u):
        x_min = t_u[i_u]
        x_max = t_u[i_u+1]
        dx = x_max - x_min
        for i_v in range(0, n_elements_v):
            y_min = t_v[i_v]
            y_max = t_v[i_v+1]
            dy = y_max - y_min

            contribution = 0.0
            for j_u in range(0, p_u+1):
                x = x_min + dx * us[j_u]
                for j_v in range(0, p_v+1):
                    y = y_min + dy * vs[j_v]

                    w = wus[j_u] * wvs[j_v]
                    w = 0.5 * dx * dy * w

                    f = f_scalar(x,y)
                    contribution = contribution + w * f
            r[i] = contribution
            i = i + 1
    return r

n_elements_u = 2
n_elements_v = 2
p_u = 2
p_v = 2
n_u = p_u + n_elements_u
n_v = p_v + n_elements_v

knots_u    = make_knots(n_u, p_u)
knots_v    = make_knots(n_v, p_v)
greville_u = make_greville(knots_u, n_u, p_u)
greville_v = make_greville(knots_v, n_v, p_v)

#print("knots_u = ", knots_u)
#print("knots_v = ", knots_v)
#print("greville_u = ", greville_u)
#print("greville_v = ", greville_v)

r_0 = interpolate_V_0(greville_u, greville_v, n_u, n_v, p_u, p_v)
r_1 = interpolate_V_1(greville_u, greville_v, n_u, n_v, p_u, p_v)
r_2 = interpolate_V_2(greville_u, greville_v, n_u, n_v, p_u, p_v)
r_3 = interpolate_V_3(greville_u, greville_v, n_u, n_v, p_u, p_v)
print("r_0 = ", r_0)
print("r_1 = ", r_1)
print("r_2 = ", r_2)
print("r_3 = ", r_3)
