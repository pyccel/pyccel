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
# ...

# ... construct the quadrature points grid
for i_element in range(0, n_elements_2):
    a = grid_2[i_element]
    b = grid_2[i_element+1]
    half = (b - a)/2.0

    for i_point in range(0, k2):
        points_2 [i_point, i_element] = a + (1.0 + u2[i_point]) * half
        weights_2[i_point, i_element] = half * w2[i_point]
# ...

# ...
print("> points_1 = ", points_1)
print("> points_2 = ", points_2)
# ...



## number of derivatives
#d1 = 2
#d2 = 2
#
#dN1 = zeros((p1+1,d1+1,p1+1), double)
#dN1 = spl_eval_splines_ders(p1, n1, d1, p1, knots1, u1)
