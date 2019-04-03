import numpy as np
import time

from pyccel.decorators import types, pure, shapes
from pyccel.epyccel import epyccel
from pyccel.epyccel import lambdify

# TODO must be known in pyccel
from operator import add, mul

#=========================================================
#VERBOSE = True
VERBOSE = False

#ACCEL = 'openmp'
ACCEL = None

_lambdify = lambda g: lambdify( g,
                                accelerator = ACCEL,
                                verbose     = VERBOSE,
                                namespace   = globals() )
#=========================================================

#=========================================================
@pure
@shapes(r='n')
@types('int', 'real', 'real[:]')
def f(n, x, r):
    for i in range(0, n):
        r[i] = i*x

#=========================================================
def test_map_1():
    g = lambda n, xs: [f(n, x, r) for x in xs]

    g = _lambdify(g)

    n = 500
    rs = np.zeros(n, np.float64)

    nx = 5000
    xs = np.linspace(0., 1., nx)

    tb = time.time()
    g(xs, n, rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#########################################
if __name__ == '__main__':
#    _f = epyccel(f)
#    n = 500
#    rs = np.zeros(n, np.float64)
#    _f(1.,n,rs)

    test_map_1()
