import numpy as np

# ==============================================================================
def blas_dnrm2(x: 'float64[:]',
               incx: 'int64' = 1,
              ):
    """
    DNRM2 returns the euclidean norm of a vector via the function
    name, so that

        DNRM2 := sqrt( x'*x )
        ||x||_2
    """
    from pyccel.stdlib.internal.blas import dnrm2

    n = x.shape[0]

    return dnrm2 (n, x, incx)
