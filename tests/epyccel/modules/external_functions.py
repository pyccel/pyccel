# pylint: disable=missing-function-docstring, missing-module-docstring/

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

# ==============================================================================
def blas_dasum(x: 'float64[:]',
               incx: 'int64' = 1,
              ):
    """
    asum ← ||re(x)||_1 + ||im(x)||_1
    """
    from pyccel.stdlib.internal.blas import dasum

    n = x.shape[0]

    return dasum (n, x, incx)

# ==============================================================================
def blas_idamax(x: 'float64[:]',
               incx: 'int64' = 1,
              ):
    """
    amax ← 1 st k ∋ |re(x k )| + |im(x k )|
            = max(|re(x i )| + |im(x i )|)
    """
    from pyccel.stdlib.internal.blas import idamax

    n = x.shape[0]

    i = idamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
def blas_ddot(x: 'float64[:]', y: 'float64[:]',
               incx: 'int64' = 1,
               incy: 'int64' = 1
              ):
    """
    y ← x
    """
    from pyccel.stdlib.internal.blas import ddot

    n = x.shape[0]

    return ddot (n, x, incx, y, incy)
