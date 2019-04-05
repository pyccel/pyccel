# TODO
#g = lambda xs,ys,z: [[x + y*z for x in xs] for y in ys]
#g = lambda xs,y,z: [x + y*z for x in xs]

import numpy as np
import time

from pyccel.decorators import types, pure
from pyccel.epyccel import epyccel
from pyccel.epyccel import lambdify
from pyccel.functional import add, mul
from pyccel.functional import where

#=========================================================
#VERBOSE = True
VERBOSE = False

#ACCEL = 'openmp'
ACCEL = None

#_lambdify = lambda g: lambdify( g, *args,
#                                accelerator = ACCEL,
#                                verbose     = VERBOSE,
#                                namespace   = globals() )
#=========================================================

#=========================================================
@pure
@types('int', 'int')
def f(x,y):
    r = x+y
    return r

# ...
@pure
@types('int')
def incr(x):
    r = x+1
    return r
# ...

#=========================================================
def test_where_1():
    g = lambda xs: [f(x,y) for x in xs]

    g = lambdify(g, where(y=1),
                 accelerator=ACCEL,
                 verbose=VERBOSE,
                 namespace=globals())

    nx = 500
    xs = range(0, nx)
    rs = np.zeros(nx, np.int32)

    tb = time.time()
    g(xs, rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_where_2():
    g = lambda xs: [f(x,h(x)) for x in xs]

    g = lambdify(g, where(h=lambda x: x**2,
                          k=lambda x: x**3),
                 accelerator=ACCEL,
                 verbose=VERBOSE,
                 namespace=globals())

    nx = 500
    xs = range(0, nx)
    rs = np.zeros(nx, np.int32)

    tb = time.time()
    g(xs, rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#=========================================================
def test_where_3():

    g = lambda xs: [f(k(x),h(x)) for x in xs]

    g = lambdify(g, where(h=lambda x: x**2,
                          k=lambda x: incr(x)),
                 accelerator=ACCEL,
                 verbose=VERBOSE,
                 namespace=globals())

    nx = 500
    xs = range(0, nx)
    rs = np.zeros(nx, np.int32)

    tb = time.time()
    g(xs, rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)

#########################################
if __name__ == '__main__':
    test_where_1()
    test_where_2()
    test_where_3()
