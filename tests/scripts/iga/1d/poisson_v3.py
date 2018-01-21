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
n_elements_1 = 4
p1 = 3

# number of derivatives
d1 = 1

n1 = p1 + n_elements_1
k1 = p1 + 1

verbose = False
#verbose = True
# ...

# ...
[u1,w1] = legendre(p1)
# ...

# ...
m1 = n1 + p1 + 1
knots1 = zeros(m1, double)
# call to spl
knots1 = spl_make_open_knots (n1, p1)
# ...

# ... TODO fix args of zeros
m1 = n_elements_1+1
grid_1 = zeros(m1, double)

# call to spl
grid_1 = spl_construct_grid_from_knots(p1, n1, n_elements_1, knots1)
# ...

# ... construct the quadrature points grid
points_1  = zeros((k1, n_elements_1), double)
weights_1 = zeros((k1, n_elements_1), double)

# call to spl
[points_1, weights_1] = spl_construct_quadrature_grid(u1, w1, grid_1)
# ...

# ...
basis_1  = zeros((p1+1, d1+1, k1, n_elements_1), double)

# call to spl
basis_1 = spl_eval_on_grid_splines_ders(n1, p1, d1, knots1, points_1)
# ...

# ...
spans_1 = zeros(n_elements_1, int)
spans_1 = spl_compute_spans(p1, n1, knots1)
# ...

# ...
start_1 = 0
end_1   = n1-1
pad_1   = p1
# ...

# ...
mass      = stencil(start_1, end_1, pad_1)
stiffness = stencil(start_1, end_1, pad_1)
rhs       = vector(start_1-pad_1, end_1+pad_1)
# ...

# ... build matrix
for ie1 in range(0, n_elements_1):
    i_span_1 = spans_1[ie1]
    for il_1 in range(0, p1+1):
        for jl_1 in range(0, p1+1):
            i1 = i_span_1 - p1  - 1 + il_1
            j1 = i_span_1 - p1  - 1 + jl_1

            v_m = 0.0
            v_s = 0.0
            for g1 in range(0, k1):
                bi_0 = basis_1[il_1, 0, g1, ie1]
                bi_x = basis_1[il_1, 1, g1, ie1]

                bj_0 = basis_1[jl_1, 0, g1, ie1]
                bj_x = basis_1[jl_1, 1, g1, ie1]

                wvol = weights_1[g1, ie1]

                v_m += bi_0 * bj_0 * wvol
                v_s += bi_x * bj_x * wvol

            mass[j1 - i1, i1] += v_m
            stiffness[j1 - i1, i1] += v_s
# ...

# ... build rhs
for ie1 in range(0, n_elements_1):
    i_span_1 = spans_1[ie1]
    for il_1 in range(0, p1+1):
        i1 = i_span_1 - p1  - 1 + il_1

        v = 0.0
        for g1 in range(0, k1):
            bi_0 = basis_1[il_1, 0, g1, ie1]
            bi_x = basis_1[il_1, 1, g1, ie1]

            x    = points_1[g1, ie1]
            wvol = weights_1[g1, ie1]

            v += bi_0 * x * (1.0 - x) * wvol

        rhs[i1] += v
# ...

# ... define matrix-vector product
#$ header procedure mv(double [:,:], double [:], double [:])
def mv(mat, x, y):
    y = 0.0
    for i in range(start_1, end_1+1):
        for k in range(-p1, p1+1):
            j = k+i
            y[i] = y[i] + mat[k, i] * x[j]
# ...

# ... CGL performs maxit CG iterations on the linear system Ax = b
#     starting from x = x0

#$ header procedure cgl(double [:,:], double [:], double [:], int, double)
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
    rdr = dot(r,r)
    for i_iter in range(1, maxit+1):
        mv(mat, p, q)
        alpha = rdr / dot (p, q)
        xk = xk + alpha * p
        r  = r - alpha * q

        norm_err = sqrt(dot(r, r))
        print (i_iter, norm_err)
        if norm_err < tol:
            x0 = xk
            break

        rdrold = rdr
        rdr = dot(r, r)
        beta = rdr / rdrold
        p = r + beta * p

    x0 = xk
# ...

# ... CRL performs maxit CG iterations on the linear system Ax = b
#     where A is a symmetric positive definite matrix, using CG method
#     starting from x = x0

#$ header procedure crl(double [:,:], double [:], double [:], int, double)
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
    sdr = dot(s,r)
    for i_iter in range(1, maxit+1):
        alpha = sdr / dot (q, q)
        xk = xk + alpha * p
        r  = r - alpha * q

        norm_err = sqrt(dot(r, r))
        print (i_iter, norm_err)
        if norm_err < tol:
            x0 = xk
            break

        mv(mat, r, s)
        sdrold = sdr
        sdr = dot(s, r)
        beta = sdr / sdrold
        p = r + beta * p
        q = s + beta * q

    x0 = xk
# ...

# ...
x0 = vector(start_1-pad_1, end_1+pad_1)
xn = vector(start_1-pad_1, end_1+pad_1)
y  = vector(start_1-pad_1, end_1+pad_1)
# ...

# ...
n_maxiter = 100
tol = 1.0e-7

xn = 0.0
cgl(mass, rhs, xn, n_maxiter, tol)

xn = 0.0
crl(mass, rhs, xn, n_maxiter, tol)

mv(mass, xn, x0)
print ('> residual error = ', max(abs(x0-rhs)))
# ...


del knots1
del grid_1
del points_1
del weights_1
del basis_1
del spans_1
del mass
del stiffness
del rhs
