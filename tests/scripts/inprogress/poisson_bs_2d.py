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
basis_1  = zeros((d1+1, k1, k1, n_elements_1), double)
basis_2  = zeros((d1+1, k2, k2, n_elements_2), double)

dN1 = zeros((p1+1, d1+1, k1), double)
dN2 = zeros((p2+1, d2+1, k2), double)
# ...

# ... evaluates B-Splines and their derivatives on the quad grid
for i_element in range(0, n_elements_1):
    dN1 = 0.0
    dN1 = spl_eval_splines_ders(p1, n1, d1, p1, knots1, u1)
    basis_1[:,:,:,i_element] = dN1

for i_element in range(0, n_elements_2):
    dN2 = 0.0
    dN2 = spl_eval_splines_ders(p2, n2, d2, p2, knots2, u2)
    basis_2[:,:,:,i_element] = dN2
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
del dN1
del dN2
