import numpy as np

from pyccel.decorators import types
from pyccel.functional import lambdify
from pyccel.epyccel import epyccel

@types('int', 'int', 'int')
def f(x,y,z):
    r = x+y*z
    return r

g = lambdify('map', f, rank=(1,1,0))

#VERBOSE = True
VERBOSE = False

#g = epyccel(g, accelerator='openmp', verbose=VERBOSE)
mod = epyccel(g, verbose=VERBOSE)

nx = 5
ny = 4
arr_x = range(0, nx)
arr_y = range(0, ny)
arr_r = np.zeros((nx,ny), np.int32)

mod.map_f(arr_x, arr_y, 2, arr_r)
print(arr_r)
