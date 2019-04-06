# TODO
#g = lambda xs,ys,z: [[x + y*z for x in xs] for y in ys]
#g = lambda xs,y,z: [x + y*z for x in xs]

import numpy as np
import time

from pyccel.decorators import types, pure
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

@pure
@types('int')
def square(x):
    r = x**2
    return r

lam_1 = lambda xs: [square(x) for x in xs]
lam_2 = lambda x: x**2 + 3*x


#==============================================================================
#      INTEGER CASE
#==============================================================================

#=========================================================
def test_map_int_1():
    L = lambda xs,ys,z: [f_int(lam_1(x),lam_2(y),z) for x in xs for y in ys]

    L = lambdify( L,
                  namespace   = {'f_int': f_int,
                                 'lam_2': lam_2,
                                 'lam_1': lam_1,
                                 'square': square},
                  accelerator = ACCEL,
                  verbose     = VERBOSE )

    nx = 5000
    ny = 4000
    xs = range(0, nx)
    ys = range(0, ny)
    rs = np.zeros(nx*ny, np.int32)

    tb = time.time()
    L(xs, ys, 2, rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)


#########################################
if __name__ == '__main__':
    # ... int examples
    test_map_int_1()
