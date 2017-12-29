# coding: utf-8

For = load('pyccel.ast.core', 'For', False, 3)
dx = load('pyccel.symbolic.gelato', 'dx', False, 1)
dy = load('pyccel.symbolic.gelato', 'dy', False, 1)

a  = lambda u,v: dx(u) * dx(v) + dy(u) * dy(v)

weak_form = lambdify(a)

#$ header function kernel(int, int, double [:], double [:], double [:], double [:], double [:,:], double [:,:])
def kernel(p1, p2, u, v, wu, wv, bi, bj):
    mat = zeros((p1+1,p2+1))

    for i in range(0, p1+1):
        for j in range(0, p2+1):
            mat[i,j] = mat[i,j] + weak_form(u[i], v[j]) * wu[i] * wv[j]
