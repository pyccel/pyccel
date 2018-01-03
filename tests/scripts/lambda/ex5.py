# coding: utf-8

# TODO: - why must we put 'g1, g2 = 0'? => problem with local_vars/stmt in syntax?
#       - improve how we pass arguments to weak_form

# ... import symbolic tools
weak_formulation = load('pyccel.symbolic.gelato', 'weak_formulation', True, 2)
dx               = load('pyccel.symbolic.gelato', 'dx', False, 1)
dy               = load('pyccel.symbolic.gelato', 'dy', False, 1)
# ...

# ... weak formulation
a  = lambda x,y,u,v: dx(u) * dx(v) + dy(u) * dy(v)
wf = weak_formulation(a, 2)

weak_form = lambdify(wf)
# ...

#$ header function kernel(int, int, double [:], double [:], double [:], double [:], double [:,:], double [:,:], double [:,:], double [:,:]) results (double)
def kernel(p1, p2, u, v, wu, wv, bi1, bi2, bj1, bj2):
    contrib = 0.0

    val  = 0.0

    x    = 0.0
    y    = 0.0

    Ni   = 0.0
    Ni_x = 0.0
    Ni_y = 0.0
    Nj   = 0.0
    Nj_x = 0.0
    Nj_y = 0.0

    g1 = 0
    g2 = 0

    for g1 in range(0, p1+1):
        for g2 in range(0, p2+1):

            x = u[g1]
            y = v[g2]

            Ni   = bi1[0,g1] * bi2[0,g2]
            Ni_x = bi1[1,g1] * bi2[0,g2]
            Ni_y = bi1[0,g1] * bi2[1,g2]

            Nj   = bj1[0,g1] * bj2[0,g2]
            Nj_x = bj1[1,g1] * bj2[0,g2]
            Nj_y = bj1[0,g1] * bj2[1,g2]

            val = weak_form(x, y, Ni_x, Ni_y, Nj_x, Nj_y)
            contrib = contrib + val * wu[g1] * wv[g2]

    return contrib
