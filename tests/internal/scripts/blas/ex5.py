# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np

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

