# coding: utf-8

# - run the command:
#    > pyccel-quickstart poisson
#   this will compile pyccel extensions and install them in $PWD/poisson/usr
# - export the following variables
#    > export INCLUDE_DIR=$PWD/poisson/usr/include/poisson
#    > export LIB_DIR=$PWD/poisson/usr/lib

# Usage:
#    > pyccel poisson_bs_2d.py --include='$INCLUDE_DIR' --libdir='$LIB_DIR' --libs=poisson --no-modules

# Cleaning:
#    > rm -f *.mod *.pyccel *.f90 *.o



from pyccelext.math.bsplines     import make_knots
from pyccelext.math.bsplines     import make_greville
from pyccelext.math.quadratures  import legendre
from pyccelext.math.external.bsp import spl_eval_splines_ders
from pyccelext.math.external.bsp import spl_compute_spans


# ...
n_elements_1 = 4
n_elements_2 = 4

p1 = 3
p2 = 3

n1 = p1 + n_elements_1
n2 = p2 + n_elements_2

k1 = p1+1
k2 = p2+1

# number of derivatives
d1 = 1
d2 = 1

verbose = False
# ...

# ...
[u1,w1] = legendre(p1)
# ...

# ...
[u2,w2] = legendre(p2)
# ...

# ...
knots1 = make_knots (n1, p1)
knots2 = make_knots (n2, p2)

if verbose:
    print("> knots1 = ", knots1)
    print("> knots2 = ", knots2)
# ...

# ... TODO fix args of zeros
m1 = n_elements_1+1
m2 = n_elements_2+1

grid_1 = zeros(m1, double)
grid_2 = zeros(m2, double)

for i in range(0, n_elements_1 + 1):
    grid_1[i] = knots1[i+p1]

for i in range(0, n_elements_2+1):
    grid_2[i] = knots2[i+p2]

if verbose:
    print("> grid_1 = ", grid_1)
    print("> grid_2 = ", grid_2)
# ...

# ...
points_1  = zeros((k1, n_elements_1), double)
weights_1 = zeros((k1, n_elements_1), double)

points_2  = zeros((k2, n_elements_2), double)
weights_2 = zeros((k2, n_elements_2), double)
# ...

# ... construct the quadrature points grid
for i_element in range(0, n_elements_1):
    a = grid_1[i_element]
    b = grid_1[i_element+1]
    half = (b - a)/2.0

    for i_point in range(0, k1):
        points_1 [i_point, i_element] = a + (1.0 + u1[i_point]) * half
        weights_1[i_point, i_element] = half * w1[i_point]

for i_element in range(0, n_elements_2):
    a = grid_2[i_element]
    b = grid_2[i_element+1]
    half = (b - a)/2.0

    for i_point in range(0, k2):
        points_2 [i_point, i_element] = a + (1.0 + u2[i_point]) * half
        weights_2[i_point, i_element] = half * w2[i_point]
# ...

# ...
if verbose:
    print("> points_1 = ", points_1)
    print("> points_2 = ", points_2)
# ...

# ...
basis_1  = zeros((p1+1, d1+1, k1, n_elements_1), double)
basis_2  = zeros((p2+1, d1+1, k2, n_elements_2), double)
# ...

# ... evaluates B-Splines and their derivatives on the quad grid
dN1 = zeros((p1+1, d1+1, k1), double)
for i_element in range(0, n_elements_1):
    dN1 = 0.0
    dN1 = spl_eval_splines_ders(p1, n1, d1, p1, knots1, u1)
    basis_1[:,:,:,i_element] = dN1
del dN1

dN2 = zeros((p2+1, d2+1, k2), double)
for i_element in range(0, n_elements_2):
    dN2 = 0.0
    dN2 = spl_eval_splines_ders(p2, n2, d2, p2, knots2, u2)
    basis_2[:,:,:,i_element] = dN2
del dN2
# ...

# ...
spans_1 = zeros(n_elements_1, int)
spans_2 = zeros(n_elements_2, int)

basis_elements_1 = zeros(n1, int)
basis_elements_2 = zeros(n2, int)
# ...

# subroutine call
[spans_1, basis_elements_1] = spl_compute_spans(p1, n1, knots1)

# subroutine call
[spans_2, basis_elements_2] = spl_compute_spans(p2, n2, knots2)
# ...

# ...
for i1 in range(0, p1):
    for i2 in range(0, p2):
        for m1 in range(-p1, p1):
            for m2 in range(-p2, p2):
                j1  = 0
                j2  = 0

                ie1 = 0
                ie2 = 0

                v = 0.0
                for g1 in range(0, k1):
                    for g2 in range(0, k2):
                        bi1_0 = basis_1[i1, 0, g1, ie1]
                        bi1_s = basis_1[i1, 1, g1, ie1]
                        bi2_0 = basis_2[i2, 0, g2, ie2]
                        bi2_s = basis_2[i2, 2, g2, ie2]

                        bj1_0 = basis_1[j1, 0, g1, ie1]
                        bj1_s = basis_1[j1, 1, g1, ie1]
                        bj2_0 = basis_2[j2, 0, g2, ie2]
                        bj2_s = basis_2[j2, 2, g2, ie2]

                        bi_0  = bi1_0 * bi2_0
                        bi_x  = bi1_s * bi2_0
                        bi_y  = bi1_0 * bi2_s

                        bj_0  = bj1_0 * bj2_0
                        bj_x  = bj1_s * bj2_0
                        bj_y  = bj1_0 * bj2_s

                        wvol = weights_1[g1, ie1] * weights_2[g2, ie2]

                        v += (bi_0 * bj_0 + bi_x * bj_x + bi_y * bj_y) * wvol

# ...

# ...
if verbose:
    print("> basis_1 = ", basis_1)
    print("> basis_2 = ", basis_2)
# ...

del knots1
del knots2
del grid_1
del grid_2
del points_1
del weights_1
del points_2
del weights_2
del basis_1
del basis_2
