# TODO add int examples

import numpy as np
import time

from pyccel.decorators import types, pure, shapes, workplace
from pyccel.epyccel import epyccel
from pyccel.epyccel import lambdify

# TODO must be known in pyccel
from operator import add, mul

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
@shapes(ws='n', rs='n')
@workplace('ws')
@types('int', 'real', 'real', 'real[:]', 'real[:]')
def f(n, x, y, ws, rs):
    for i in range(0, n):
        ws[i] = i*x + y

    rs[:] = ws[:]

#=========================================================
def test_map_real_1():
    g = lambda n, xs, y: [f(n, x, y, tmp, zs) for x in xs]

    g = _lambdify(g)

    nx = 10
    xs = np.linspace(0., 1., nx)

    tb = time.time()

    n = 5
    zs = np.zeros((nx,n), dtype=np.float64)
    g(n, xs, 1., zs)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_map_real_2():
    g = lambda n, xs, ys: [f(n, x, y, tmp, zs) for x in xs for y in ys]

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()

    n = 5
    zs = np.zeros((nx*ny, n), dtype=np.float64)
    g(n, xs, ys, zs)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_map_real_3():
    g = lambda n, xs, ys: [[f(n, x, y, tmp, zs) for x in xs] for y in ys]

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()

    n = 5
    zs = np.zeros((nx,ny,n), dtype=np.float64)
    g(n, xs, ys, zs)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_real_1():
    g = lambda n, xs, y: sum([f(n, x, y, tmp, rs) for x in xs])

    g = _lambdify(g)

    nx = 10
    xs = np.linspace(0., 1., nx)

    tb = time.time()

    n = 5
    g(n, xs, 1.)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_real_2():
    g = lambda n, xs, ys: sum([f(n, x, y, tmp, rs) for x in xs for y in ys])

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()

    n = 5
    g(n, xs, ys)

    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_real_3():
    g = lambda n, xs, ys: sum([[f(n, x, y, tmp, rs) for x in xs] for y in ys])

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()

    n = 5
    g(n, xs, ys)

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
