import numpy as np

from pyccel.decorators import types, pure
from pyccel.functional import lambdify
from pyccel.epyccel import epyccel

@pure
@types('int', 'int', 'int')
def f(x,y,z):
    r = x+y*z
    return r

g = lambdify('map', f, rank=(1,1,0))

#VERBOSE = True
VERBOSE = False

mod = epyccel(g, accelerator='openmp', verbose=VERBOSE)
#mod = epyccel(g, verbose=VERBOSE)

nx = 5000
ny = 4000
arr_x = range(0, nx)
arr_y = range(0, ny)
arr_r = np.zeros((nx,ny), np.int32)

import time
tb = time.time()
mod.map_f(arr_x, arr_y, 2, arr_r)
te = time.time()
print('> Elapsed time = ', te-tb)
#print(arr_r)
