# coding: utf-8

# - run the command:
#    > pyccel-quickstart poisson
#   this will compile pyccel extensions and install them in $PWD/poisson/usr
# - export the following variables
#    > export INCLUDE_DIR=$PWD/poisson/usr/include/poisson
#    > export LIB_DIR=$PWD/poisson/usr/lib

# Usage:
#    > pyccel poisson_bs_1d.py --include='$INCLUDE_DIR' --libdir='$LIB_DIR' --libs=poisson --no-modules

# Cleaning:
#    > rm -f *.mod *.pyccel *.f90 *.o



from pyccelext.math.bsplines     import make_knots
from pyccelext.math.bsplines     import make_greville
from pyccelext.math.quadratures  import legendre
from pyccelext.math.external.bsp import spl_eval_splines_ders
from pyccelext.math.external.bsp import spl_compute_spans

# ... TODO import from pyccelext.math.constants
pi = 3.141592653589793
# ...

# ...
n_elements_1 = 4

p1 = 3

n1 = p1 + n_elements_1

k1 = p1+1

# number of derivatives
d1 = 1

verbose = False
#verbose = True
# ...

# ...
[u1,w1] = legendre(p1)
# ...

# ...
knots1 = make_knots (n1, p1)

if verbose:
    print("> knots1 = ", knots1)
# ...

# ... TODO fix args of zeros
m1 = n_elements_1+1

grid_1 = zeros(m1, double)

for i in range(0, n_elements_1 + 1):
    grid_1[i] = knots1[i+p1]


if verbose:
    print("> grid_1 = ", grid_1)
# ...

# ...
points_1  = zeros((k1, n_elements_1), double)
weights_1 = zeros((k1, n_elements_1), double)
# ...

# ... construct the quadrature points grid
for i_element in range(0, n_elements_1):
    a = grid_1[i_element]
    b = grid_1[i_element+1]
    half = (b - a)/2.0

    for i_point in range(0, k1):
        points_1 [i_point, i_element] = a + (1.0 + u1[i_point]) * half
        weights_1[i_point, i_element] = half * w1[i_point]
# ...

# ...
if verbose:
    print("> points_1 = ", points_1)
# ...

# ...
basis_1  = zeros((p1+1, d1+1, k1, n_elements_1), double)
# ...

# ... evaluates B-Splines and their derivatives on the quad grid
dN1 = zeros((p1+1, d1+1, k1), double)
for i_element in range(0, n_elements_1):
    dN1 = 0.0
    dN1 = spl_eval_splines_ders(p1, n1, d1, p1, knots1, u1)
    basis_1[:,:,:,i_element] = dN1
del dN1
# ...

# ...
spans_1 = zeros(n_elements_1, int)
spans_1 = spl_compute_spans(p1, n1, knots1)
# ...

# ...
if verbose:
    print ('> spans_1 = ', spans_1)
# ...

# ...
start_1 = 0
end_1   = n1-1
pad_1   = p1

matrix  = stencil(start_1, end_1, pad_1)
rhs     = vector(start_1-pad_1, end_1+pad_1)
# ...

# ... build matrix
for ie1 in range(0, n_elements_1):
    i_span_1 = spans_1[ie1]
    for il_1 in range(0, p1+1):
        for jl_1 in range(0, p1+1):
            i1 = i_span_1 - p1  - 1 + il_1
            j1 = i_span_1 - p1  - 1 + jl_1

            v = 0.0
            for g1 in range(0, k1):
                bi_0 = basis_1[il_1, 0, g1, ie1]
                bi_x = basis_1[il_1, 1, g1, ie1]

                bj_0 = basis_1[jl_1, 0, g1, ie1]
                bj_x = basis_1[jl_1, 1, g1, ie1]

                wvol = weights_1[g1, ie1]

                v += (bi_0 * bj_0 + bi_x * bj_x) * wvol

            matrix[j1 - i1, i1] += v
# ...

# ... apply dirichlet boundary conditions
matrix[:,0]    = 0.0
matrix[:,n1-1] = 0.0
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

            v += (bi_0 * sin(2*pi*x)) * wvol

        rhs[i1] += v
# ...

# ...
x0 = vector(start_1-pad_1, end_1+pad_1)
y  = vector(start_1-pad_1, end_1+pad_1)

mv(matrix, rhs, y)
# ...

# ...
if verbose:
    print("> basis_1 = ", basis_1)
# ...

del knots1
del grid_1
del points_1
del weights_1
del basis_1
del spans_1
