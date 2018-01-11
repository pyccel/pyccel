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


def test_legendre():
    m = 3

    # ...
    [x,w] = legendre(m)
    # ...

    print(x)
    print(w)


def test_2():
    n_elements = 4
    p = 2
    n = p+n_elements

    knots = make_knots(n, p)
    print(" knots    = ", knots)

    d = 2
    r = 3
    r1 = r + 1
    p1 = p + 1
    d1 = d + 1

    tau = zeros(r1, double)
    tau[0] = 0.1
    tau[1] = 0.3
    tau[2] = 0.7
    tau[3] = 0.8

    dN = zeros((p1,d1,r1), double)
    dN = spl_eval_splines_ders(p,n,d,r,knots,tau)

    print(" dN = ", dN)

test_2()

print('> PASSED')
