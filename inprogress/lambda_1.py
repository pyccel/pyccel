import numpy as np
import time

from pyccel.decorators import types, pure
from pyccel.epyccel import epyccel
from pyccel.epyccel import epyccel_lambda

#=========================================================
#VERBOSE = True
VERBOSE = False

ACCEL = 'openmp'
#ACCEL = None


lambdify = lambda g: epyccel_lambda( g,
                                     accelerator = ACCEL,
                                     verbose     = VERBOSE,
                                     namespace   = globals() )
#=========================================================

#=========================================================
@pure
@types('int', 'int', 'int')
def f(x,y,z):
    r = x+y*z
    return r

#g = lambda x,y,z: f(x,y,z)
#g = lambda xs,ys,z: [[x + y*z for x in xs] for y in ys]
#g = lambda xs,y,z: [x + y*z for x in xs]

#g = lambda xs,y,z: [f(x,y,z) for x in xs]


#=========================================================
def test_1():
    g = lambda xs,ys,z: [f(x,y,z) for x in xs for y in ys]

    g = lambdify(g)

    nx = 5000
    ny = 4000
    arr_x = range(0, nx)
    arr_y = range(0, ny)
    arr_r = np.zeros(nx*ny, np.int32)

    tb = time.time()
    g(arr_x, arr_y, 2, arr_r)
    te = time.time()
    print('> Elapsed time = ', te-tb)
    #print(arr_r)

#=========================================================
def test_2():
    g = lambda xs,ys,z: [[f(x,y,z) for x in xs] for y in ys]

    g = lambdify(g)

    nx = 5000
    ny = 4000
    arr_x = range(0, nx)
    arr_y = range(0, ny)
    arr_r = np.zeros((nx,ny), np.int32)

    tb = time.time()
    g(arr_x, arr_y, 2, arr_r)
    te = time.time()
    print('> Elapsed time = ', te-tb)
    #print(arr_r)

#########################################
if __name__ == '__main__':
    test_1()
    test_2()
