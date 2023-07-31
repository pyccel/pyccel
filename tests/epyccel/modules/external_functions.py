# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types


# ==============================================================================
def blas_dnrm2(x: 'float64[:]',
               incx: 'int32' = 1,
              ):
    """
    Computes the Euclidean norm of a vector.
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
    Computes the sum of magnitudes of the vector elements.
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
    Finds the index of the element with maximum absolute value.
    """
    import numpy as np
    from pyccel.stdlib.internal.blas import idamax

    n = np.int32(x.shape[0])

    i = idamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-np.int32(1)
    return i

# ==============================================================================
def blas_ddot(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    Computes a vector-vector dot product.
    """
    import numpy as np
    from pyccel.stdlib.internal.blas import ddot

    n = np.int32(x.shape[0])

    return ddot (n, x, incx, y, incy)

# ==============================================================================
def blas_ddot_in_func(x: 'float64[:]', y: 'float64[:]'):
    import numpy as np
    def blas_ddot(x: 'float64[:]', y: 'float64[:]',
                   incx: 'int32' = 1,
                   incy: 'int32' = 1
                  ):
        """
        Computes a vector-vector dot product.
        """
        from pyccel.stdlib.internal.blas import ddot

        n = np.int32(x.shape[0])

        return ddot (n, x, incx, y, incy)

    incx = np.int32(1)
    incy = np.int32(1)
    return blas_ddot(x,y,incx,incy)
