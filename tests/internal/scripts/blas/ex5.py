# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel.decorators import inline

def f(x: 'float32[:]'):
    import pyccel.stdlib.internal.blas as blas_mod
    n = np.int32(x.shape[0])
    incx = np.int32(1)
    return blas_mod.sasum (n, x, incx)

def g(vect1 : 'float64[:]', vect2 : 'float64[:]'):
    import pyccel.stdlib.internal.blas as blas_mod

    sa = 3.5
    n = np.int32(vect1.shape[0])

    blas_mod.daxpy(n, sa, vect1, np.int32(1), vect2, np.int32(1))

@inline
def sasum(x: 'float32[:]',
          incx: 'int' = 1,
         ):
    import pyccel.stdlib.internal.blas as blas_mod
    n = np.int32(x.shape[0])
    return blas_mod.sasum (n, x, np.int32(incx))

def h(x: 'float32[:]'):
    return sasum (x)
