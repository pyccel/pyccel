import numpy as np
import time

from pyccel.decorators import workplace
from pyccel.decorators import shapes
from pyccel.decorators import stack_array
from pyccel.decorators import types
from pyccel.decorators import pure
from pyccel.epyccel    import epyccel
from pyccel.epyccel    import lambdify
from pyccel.functional import add, mul
from pyccel.functional import where


#=========================================================
#VERBOSE = True
VERBOSE = False

#ACCEL = 'openmp'
ACCEL = None
#=========================================================

#==============================================================================
@shapes(values='degree+1')
@stack_array('left', 'right')
@types('double[:]','int','double','int','double[:]')
def basis_funs( knots, degree, x, span, values ):
    from numpy      import empty
    left   = empty( degree  , dtype=float )
    right  = empty( degree  , dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

#=========================================================
def test_1():
    g = lambda knots, degree, xs: [basis_funs( knots, degree, x, span, values ) for x in xs]

    g = lambdify(g, where(span=4),
                 accelerator=ACCEL,
                 verbose=VERBOSE,
                 namespace=globals())

    nx = 500
    xs = np.linspace(0., 1., nx)

    degree = 3
    grid = np.linspace(0., 1., 5)
    knots = [0.]*degree + list(grid) + [1.]*degree
    knots = np.asarray(knots)

    rs = np.zeros((nx,degree+1), np.float64)

    tb = time.time()
    g(knots, degree, xs, rs)
    te = time.time()
    print('> Elapsed time = ', te-tb)


#########################################
if __name__ == '__main__':
    test_1()
