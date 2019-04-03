# TODO
#g = lambda xs,ys,z: [[x + y*z for x in xs] for y in ys]
#g = lambda xs,y,z: [x + y*z for x in xs]

import numpy as np
import time

from pyccel.decorators import types, pure
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
@pure
@types('int', 'int', 'int')
def f_int(x,y,z):
    r = x+y*z
    return r

@pure
@types('real', 'real', 'real')
def f_real(x,y,z):
    r = x+y*z
    return r


#==============================================================================
#      INTEGER CASE
#==============================================================================

#=========================================================
def test_map_int_1():
    g = lambda xs,ys,z: [f_int(x,y,z) for x in xs for y in ys]

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = range(0, nx)
    ys = range(0, ny)
    rs = np.zeros(nx*ny, np.int32)

    tb = time.time()
    g(xs, ys, 2, rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_map_int_2():
    g = lambda xs,ys,z: [[f_int(x,y,z) for x in xs] for y in ys]

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = range(0, nx)
    ys = range(0, ny)
    rs = np.zeros((nx,ny), np.int32)

    tb = time.time()
    g(xs, ys, 2, rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_int_1():
    g = lambda xs,y,z: add([f_int(x,y,z) for x in xs])

    g = _lambdify(g)

    nx = 5000
    xs = range(0, nx)

    tb = time.time()
    g(xs, 3, 2)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_int_2():
    g = lambda xs,ys,z: add([[f_int(x,y,z) for x in xs] for y in ys])

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = range(0, nx)
    ys = range(0, ny)

    tb = time.time()
    g(xs, ys, 2)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_mul_map_int_1():
    g = lambda xs,y,z: mul([f_int(x,y,z) for x in xs])

    g = _lambdify(g)

    nx = 5000
    xs = range(0, nx)

    tb = time.time()
    g(xs, 3, 2)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_mul_map_int_2():
    g = lambda xs,ys,z: mul([[f_int(x,y,z) for x in xs] for y in ys])

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = range(0, nx)
    ys = range(0, ny)

    tb = time.time()
    g(xs, ys, 2)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#==============================================================================
#      REAL CASE
#==============================================================================

#=========================================================
def test_map_real_1():
    g = lambda xs,ys,z: [f_real(x,y,z) for x in xs for y in ys]

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)
    rs = np.zeros(nx*ny, np.float64)

    tb = time.time()
    g(xs, ys, 2., rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_map_real_2():
    g = lambda xs,ys,z: [[f_real(x,y,z) for x in xs] for y in ys]

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)
    rs = np.zeros((nx,ny), np.float64)

    tb = time.time()
    g(xs, ys, 2., rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_real_1():
    g = lambda xs,y,z: add([f_real(x,y,z) for x in xs])

    g = _lambdify(g)

    nx = 5000
    xs = np.linspace(0., 1., nx)

    tb = time.time()
    g(xs, 3., 2.)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_add_map_real_2():
    g = lambda xs,ys,z: add([[f_real(x,y,z) for x in xs] for y in ys])

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()
    g(xs, ys, 2.)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_mul_map_real_1():
    g = lambda xs,y,z: mul([f_real(x,y,z) for x in xs])

    g = _lambdify(g)

    nx = 5000
    xs = np.linspace(0., 1., nx)

    tb = time.time()
    g(xs, 3., 2.)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_mul_map_real_2():
    g = lambda xs,ys,z: mul([[f_real(x,y,z) for x in xs] for y in ys])

    g = _lambdify(g)

    nx = 5000
    ny = 4000
    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)

    tb = time.time()
    g(xs, ys, 2.)
    te = time.time()
    print('> Elapsed time = ', te-tb)


#########################################
if __name__ == '__main__':
    # ... int examples
    test_map_int_1()
    test_map_int_2()
    test_add_map_int_1()
    test_add_map_int_2()
    test_mul_map_int_1()
    test_mul_map_int_2()
    # ...

    # ... real examples
    test_map_real_1()
    test_map_real_2()
    test_add_map_real_1()
    test_add_map_real_2()
    test_mul_map_real_1()
    test_mul_map_real_2()
    # ...
