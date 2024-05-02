# coding: utf-8

# - run the command:
#    > pyccel-quickstart poisson
#   this will compile pyccel extensions and install them in $PWD/poisson/usr
# - export the following variables
#    > export INCLUDE_DIR=$PWD/poisson/usr/include/poisson
#    > export LIB_DIR=$PWD/poisson/usr/lib

# Usage:
#    > pyccel poisson_v2.py --include='$INCLUDE_DIR' --libdir='$LIB_DIR' --libs=poisson --no-modules --execute

# Cleaning:
#    > rm -f *.mod *.pyccel *.f90 *.o


from pyccelext.math.quadratures  import legendre
from pyccelext.math.external.bsp import spl_make_open_knots
from pyccelext.math.external.bsp import spl_compute_spans
from pyccelext.math.external.bsp import spl_construct_grid_from_knots
from pyccelext.math.external.bsp import spl_construct_quadrature_grid
from pyccelext.math.external.bsp import spl_eval_on_grid_splines_ders


# ...
p1 = 2
p2 = 2

n_elements_1 = 8
n_elements_2 = 8

n_elements_1 = n_elements_1 - p1
n_elements_2 = n_elements_2 - p2

# number of derivatives
d1 = 1
d2 = 1

n1 = p1 + n_elements_1
n2 = p2 + n_elements_2

k1 = p1 + 1
k2 = p2 + 1

verbose = False
#verbose = True
# ...

# ...
[u1,w1] = legendre(p1)
# ...

# ...
[u2,w2] = legendre(p2)
# ...

# ...
m1 = n1 + p1 + 1
m2 = n2 + p2 + 1

knots1 = zeros(m1, double)
knots2 = zeros(m2, double)

# call to spl
knots1 = spl_make_open_knots (n1, p1)

# call to spl
knots2 = spl_make_open_knots (n2, p2)
# ...

# ... TODO fix args of zeros
m1 = n_elements_1+1
m2 = n_elements_2+1

grid_1 = zeros(m1, double)
grid_2 = zeros(m2, double)

# call to spl
grid_1 = spl_construct_grid_from_knots(p1, n1, n_elements_1, knots1)

# call to spl
grid_2 = spl_construct_grid_from_knots(p2, n2, n_elements_2, knots2)
# ...

# ... construct the quadrature points grid
points_1  = zeros((k1, n_elements_1), double)
points_2  = zeros((k2, n_elements_2), double)
weights_1 = zeros((k1, n_elements_1), double)
weights_2 = zeros((k2, n_elements_2), double)

# call to spl
[points_1, weights_1] = spl_construct_quadrature_grid(u1, w1, grid_1)

# call to spl
[points_2, weights_2] = spl_construct_quadrature_grid(u2, w2, grid_2)
# ...

# ...
basis_1  = zeros((p1+1, d1+1, k1, n_elements_1), double)
basis_2  = zeros((p2+1, d2+1, k2, n_elements_2), double)

# call to spl
basis_1 = spl_eval_on_grid_splines_ders(n1, p1, d1, knots1, points_1)

# call to spl
basis_2 = spl_eval_on_grid_splines_ders(n2, p2, d2, knots2, points_2)
# ...

# ...
spans_1 = zeros(n_elements_1, int)
spans_2 = zeros(n_elements_2, int)

spans_1 = spl_compute_spans(p1, n1, knots1)
spans_2 = spl_compute_spans(p2, n2, knots2)
# ...

# ...
start_1 = 0
end_1   = n1-1
pad_1   = p1

start_2 = 0
end_2   = n2-1
pad_2   = p2
# ...

# ...
mass      = stencil((start_1, start_2), (end_1, end_2), (pad_1, pad_2))
stiffness = stencil((start_1, start_2), (end_1, end_2), (pad_1, pad_2))
rhs       = vector((start_1-pad_1, start_2-pad_2), (end_1+pad_1, end_2+pad_2))
# ...

# ... build matrix
for ie1 in range(0, n_elements_1):
    for ie2 in range(0, n_elements_2):
        i_span_1 = spans_1[ie1]
        i_span_2 = spans_2[ie2]
        for il_1 in range(0, p1+1):
            for jl_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_2 in range(0, p2+1):

                        i1 = i_span_1 - p1  - 1 + il_1
                        j1 = i_span_1 - p1  - 1 + jl_1

                        i2 = i_span_2 - p2  - 1 + il_2
                        j2 = i_span_2 - p2  - 1 + jl_2

                        v_m = 0.0
                        v_s = 0.0
                        for g1 in range(0, k1):
                            for g2 in range(0, k2):
                                bi_0 = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                bi_x = basis_1[il_1, 1, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                bi_y = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 1, g2, ie2]

                                bj_0 = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                bj_x = basis_1[jl_1, 1, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                bj_y = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 1, g2, ie2]

                                wvol = weights_1[g1, ie1] * weights_2[g2, ie2]

                                v_m += bi_0 * bj_0 * wvol
                                v_s += (bi_x * bj_x + bi_y * bj_y) * wvol

                        mass[j1 - i1, j2 - i2, i1, i2] += v_m
                        stiffness[j1 - i1, j2 - i2, i1, i2] += v_s
# ...

for i1 in range(0, n1):
    for i2 in range(0, n2):
        for k1 in range(-p1, p1+1):
            for k2 in range(-p2, p2+1):
                print (i1, i2, k1, k2, mass[k1,k2,i1,i2])

print ('done')
pass

# ... build rhs
for ie1 in range(0, n_elements_1):
    for ie2 in range(0, n_elements_2):
        i_span_1 = spans_1[ie1]
        i_span_2 = spans_2[ie2]
        for il_1 in range(0, p1+1):
            for il_2 in range(0, p2+1):
                i1 = i_span_1 - p1  - 1 + il_1
                i2 = i_span_2 - p2  - 1 + il_2

                v = 0.0
                for g1 in range(0, k1):
                    for g2 in range(0, k2):
                        bi_0 = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                        bi_x = basis_1[il_1, 1, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                        bi_y = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 1, g2, ie2]

                        x1    = points_1[g1, ie1]
                        x2    = points_2[g2, ie2]
                        wvol = weights_1[g1, ie1] * weights_2[g2, ie2]

                        v += bi_0 * x1 * (1.0 - x1) * x2 * (1.0 - x2) * wvol

                rhs[i1, i2] += v
# ...

# ... define matrix-vector product
#$ header procedure mv(double [:,:,:,:], double [:,:], double [:,:])
def mv(mat, x, y):
    y = 0.0
    for i1 in range(start_1, end_1+1):
        for i2 in range(start_2, end_2+1):
            for k1 in range(-p1, p1+1):
                for k2 in range(-p2, p2+1):
                    j1 = k1+i1
                    j2 = k2+i2
                    y[i1,i2] = y[i1,i2] + mat[k1,k2,i1,i2] * x[j1,j2]
# ...

# ... define dot for 2d arrays

#$ header function vdot(double[:,:], double[:,:]) results(double)
def vdot(xl, xr):
    r = 0.0
    for i1 in range(start_1, end_1+1):
        for i2 in range(start_2, end_2+1):
            for k1 in range(-p1, p1+1):
                for k2 in range(-p2, p2+1):
                    r += xl[k1,i1] * xr[k2,i2]
    return r
# ...

# ... CGL performs maxit CG iterations on the linear system Ax = b
#     starting from x = x0

#$ header procedure cgl(double [:,:,:,:], double [:,:], double [:,:], int, double)
def cgl(mat, b, x0, maxit, tol):
    xk = zeros_like(x0)
    mx = zeros_like(x0)
    p  = zeros_like(x0)
    q  = zeros_like(x0)
    r  = zeros_like(x0)

    xk = x0

    mv(mat, x0, mx)
    r = b - mx
    p = r
    rdr = vdot(r,r)
    for i_iter in range(1, maxit+1):
        mv(mat, p, q)
        alpha = rdr / vdot (p, q)
        xk = xk + alpha * p
        r  = r - alpha * q

        norm_err = sqrt(vdot(r, r))
        print (i_iter, norm_err)
        if norm_err < tol:
            x0 = xk
            break

        rdrold = rdr
        rdr = vdot(r, r)
        beta = rdr / rdrold
        p = r + beta * p

    x0 = xk
# ...

# ... CRL performs maxit CG iterations on the linear system Ax = b
#     where A is a symmetric positive definite matrix, using CG method
#     starting from x = x0

#$ header procedure crl(double [:,:,:,:], double [:,:], double [:,:], int, double)
def crl(mat, b, x0, maxit, tol):
    xk = zeros_like(x0)
    mx = zeros_like(x0)
    p  = zeros_like(x0)
    q  = zeros_like(x0)
    r  = zeros_like(x0)
    s  = zeros_like(x0)

    xk = x0

    mv(mat, x0, mx)
    r = b - mx
    p = r
    mv(mat, p, q)
    s = q
    sdr = vdot(s,r)
    for i_iter in range(1, maxit+1):
        alpha = sdr / vdot (q, q)
        xk = xk + alpha * p
        r  = r - alpha * q

        norm_err = sqrt(vdot(r, r))
        print (i_iter, norm_err)
        if norm_err < tol:
            x0 = xk
            break

        mv(mat, r, s)
        sdrold = sdr
        sdr = vdot(s, r)
        beta = sdr / sdrold
        p = r + beta * p
        q = s + beta * q

    x0 = xk
# ...

# ...
x0 = vector((start_1-pad_1, start_2-pad_2), (end_1+pad_1, end_2+pad_2))
xn = vector((start_1-pad_1, start_2-pad_2), (end_1+pad_1, end_2+pad_2))
y  = vector((start_1-pad_1, start_2-pad_2), (end_1+pad_1, end_2+pad_2))
# ...

# ...
n_maxiter = 100
tol = 1.0e-7

xn = 0.0
cgl(mass, rhs, xn, n_maxiter, tol)

# TODO crl is converging slowly. must be investigated
#xn = 0.0
#crl(stiffness, rhs, xn, n_maxiter, tol)

mv(mass, xn, x0)
print ('> residual error = ', max(abs(x0-rhs)))
# ...


del knots1
del grid_1
del points_1
del weights_1
del basis_1
del spans_1
del knots2
del grid_2
del points_2
del weights_2
del basis_2
del spans_2
del mass
del stiffness
del rhs
