# pylint: disable=missing-function-docstring, missing-module-docstring/


# ==============================================================================
def blas_dnrm2(x: 'float64[:]',
               incx: 'int32' = 1,
              ):
    """
    ||x||_2
    """
    import numpy as np
    from pyccel.stdlib.internal.blas import dnrm2

    n = np.int32(x.shape[0])

    return dnrm2 (n, x, incx)

# ==============================================================================
def blas_dasum(x: 'float64[:]',
               incx: 'int32' = 1,
              ):
    """
    asum ← ||re(x)|| 1 + ||im(x)|| 1
    """
    import numpy as np
    from pyccel.stdlib.internal.blas import dasum

    n = np.int32(x.shape[0])

    return dasum (n, x, incx)

# ==============================================================================
def blas_idamax(x: 'float64[:]',
               incx: 'int32' = 1,
              ):
    """
    amax ← 1 st k ∋ |re(x k )| + |im(x k )|
            = max(|re(x i )| + |im(x i )|)
    """
    import numpy as np
    from pyccel.stdlib.internal.blas import idamax

    n = np.int32(x.shape[0])

    i = idamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
def blas_ddot(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    y ← x
    """
    import numpy as np
    from pyccel.stdlib.internal.blas import ddot

    n = np.int32(x.shape[0])

    return ddot (n, x, incx, y, incy)
