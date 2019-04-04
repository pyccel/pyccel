# TODO add int examples

import numpy as np
import time

from pyccel.decorators import types, pure, shapes, workplace
from pyccel.epyccel import epyccel
from pyccel.epyccel import lambdify
from pyccel.functional import add, mul

#=========================================================
#VERBOSE = True
VERBOSE = False

ACCEL = 'openmp'
#ACCEL = None

_lambdify = lambda g: lambdify( g,
                                accelerator = ACCEL,
                                verbose     = VERBOSE,
                                namespace   = globals() )
#=========================================================

#=========================================================
#@pure
@shapes(rs='n')
@workplace('rs')
@types('int', 'real', 'real', 'real[:]')
def f(n, x, y, rs):
    for i in range(0, n):
        rs[i] = i*x + y

    z = 0.
    for i in range(0, n):
        z += rs[i]

    z = z/n
    return z


#=========================================================
def test_map_real_1():
    g = lambda n, xs, y: [f(n, x, y, rs) for x in xs]

    g = _lambdify(g)

    nx = 10
    xs = np.linspace(0., 1., nx)

    tb = time.time()

    zs = np.zeros(nx, dtype=np.float64)
    g(5, xs, 1., zs)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_map_real_2():
    g = lambda n, xs, ys: [f(n, x, y, rs) for x in xs for y in ys]

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()

    zs = np.zeros(nx*ny, dtype=np.float64)
    g(5, xs, ys, zs)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_map_real_3():
    g = lambda n, xs, ys: [[f(n, x, y, rs) for x in xs] for y in ys]

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()

    zs = np.zeros((nx,ny), dtype=np.float64)
    g(5, xs, ys, zs)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_real_1():
    g = lambda n, xs, y: sum([f(n, x, y, rs) for x in xs])

    g = _lambdify(g)

    nx = 10
    xs = np.linspace(0., 1., nx)

    tb = time.time()

    g(5, xs, 1.)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_real_2():
    g = lambda n, xs, ys: sum([f(n, x, y, rs) for x in xs for y in ys])

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()

    g(5, xs, ys)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_real_3():
    g = lambda n, xs, ys: sum([[f(n, x, y, rs) for x in xs] for y in ys])

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()

    g(5, xs, ys)

    te = time.time()
    print('> Elapsed time = ', te-tb)


#########################################
if __name__ == '__main__':
    test_map_real_1()
    test_map_real_2()
    test_map_real_3()
    test_add_map_real_1()
    test_add_map_real_2()
    test_add_map_real_3()
