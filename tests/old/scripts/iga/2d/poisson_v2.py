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
n_elements_2 = 4
p1 = 3
p2 = 3

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
