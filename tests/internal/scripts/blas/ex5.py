import numpy as np

def f(x: 'float32[:]'):
    import pyccel.stdlib.internal.blas as blas_mod
    n = np.int32(x.shape[0])
    incx = np.int32(1)
    return blas_mod.sasum (n, x, incx)
