# coding: utf-8

# - run the command:
#    > pyccel-quickstart poisson
#   this will compile pyccel extensions and install them in $PWD/poisson/usr
# - export the following variables
#    > export INCLUDE_DIR=$PWD/poisson/usr/include/poisson
#    > export LIB_DIR=$PWD/poisson/usr/lib

# Usage:
#    > pyccel poisson_v1.py --include='$INCLUDE_DIR' --libdir='$LIB_DIR' --libs=poisson --no-modules --execute

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
mass      = zeros((n1,n1), double)
stiffness = zeros((n1,n1), double)
rhs       = zeros(n1, double)
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

            mass[i1, j1] += v_m
            stiffness[i1, j1] += v_s
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
