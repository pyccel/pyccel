# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def blas_srotg(a: 'float32', b: 'float32',
               c: 'float32' = 0.,
               s: 'float32' = 0.,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    mod_blas.srotg (a, b, c, s)

    return c, s

# ==============================================================================
def blas_srotmg(d1: 'float32', d2: 'float32', x1: 'float32', y1: 'float32',
                param: 'float32[:]'):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    mod_blas.srotmg (d1, d2, x1, y1, param)

# ==============================================================================
def blas_srot(x: 'float32[:]', y: 'float32[:]', c: 'float32', s: 'float32',
              incx: 'int32' = 1,
              incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.srot (n, x, incx, y, incy, c, s)

# ==============================================================================
def blas_srotm(x: 'float32[:]', y: 'float32[:]', param: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.srotm (n, x, incx, y, incy, param)

# ==============================================================================
def blas_scopy(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.scopy (n, x, incx, y, incy)

# ==============================================================================
def blas_sswap(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.sswap (n, x, incx, y, incy)

# ==============================================================================
def blas_sscal(alpha: 'float32', x: 'float32[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.sscal (n, alpha, x, incx)

# ==============================================================================
def blas_sdot(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.sdot (n, x, incx, y, incy)

# ==============================================================================
def blas_sdsdot(sb: 'float32', x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.sdsdot (n, sb, x, incx, y, incy)

# ==============================================================================
def blas_dsdot(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.dsdot (n, x, incx, y, incy)

# ==============================================================================
def blas_saxpy(x: 'float32[:]', y: 'float32[:]',
               a: 'float32' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.saxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_snrm2(x: 'float32[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.snrm2 (n, x, incx)

# ==============================================================================
def blas_sasum(x: 'float32[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.sasum (n, x, incx)

# ==============================================================================
def blas_isamax(x: 'float32[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    i = mod_blas.isamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def blas_sgemv(alpha: 'float32', a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.sgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_sgbmv(kl : 'int32', ku: 'int32', alpha: 'float32',
               a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])

    lda = m
#    lda = np.int32(1) + ku + kl

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.sgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_ssymv(alpha: 'float32', a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.ssymv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_ssbmv(k : 'int32', alpha: 'float32',
               a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.ssbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_sspmv(alpha: 'float32', a: 'float32[:]', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.sspmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_strmv(a: 'float32[:,:](order=F)', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.strmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_stbmv(k : 'int32', a: 'float32[:,:](order=F)', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.stbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_stpmv(a: 'float32[:]', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.stpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_strsv(a: 'float32[:,:](order=F)', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.strsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_stbsv(k: 'int32', a: 'float32[:,:](order=F)', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.stbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_stpsv(a: 'float32[:]', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.stpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_sger(alpha: 'float32', x: 'float32[:]', y: 'float32[:]', a: 'float32[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.sger (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_ssyr(alpha: 'float32', x: 'float32[:]', a: 'float32[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.ssyr (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_sspr(alpha: 'float32', x: 'float32[:]', a: 'float32[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.sspr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_ssyr2(alpha: 'float32', x: 'float32[:]', y: 'float32[:]', a: 'float32[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.ssyr2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_sspr2(alpha: 'float32', x: 'float32[:]', y: 'float32[:]', a: 'float32[:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.sspr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def blas_sgemm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
               trans_a: 'bool' = False,
               trans_b: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    l = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'

    flag_trans_b = 'N'
    if trans_b: flag_trans_b = 'T'
    # ...

    # ...
    if trans_a:
        m = np.int32(a.shape[0])
    else:
        m = np.int32(a.shape[1])
    # ...

    # TODO to be checked
    lda = m
    ldb = m
    ldc = l

    mod_blas.sgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ssymm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldb = m
    ldc = m

    mod_blas.ssymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ssyrk(alpha: 'float32', a: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])
    k = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_trans = 'N'
    lda = n
    # equation 2
    if trans:
        flag_trans = 'T'
        lda = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.ssyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_ssyr2k(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])

    # ...
    # equation 1
    flag_trans = 'N'
    k = np.int32(a.shape[1])
    lda = n
    ldb = n
    # equation 2
    if trans:
        flag_trans = 'T'
        k = np.int32(a.shape[0])
        lda = k
        ldb = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.ssyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_strmm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    mod_blas.strmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_strsm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    mod_blas.strsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def blas_drotg(a: 'float64', b: 'float64',
               c: 'float64' = 0.,
               s: 'float64' = 0.,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    mod_blas.drotg (a, b, c, s)

    return c, s

# ==============================================================================
def blas_drotmg(d1: 'float64', d2: 'float64', x1: 'float64', y1: 'float64',
                param: 'float64[:]'):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    mod_blas.drotmg (d1, d2, x1, y1, param)

# ==============================================================================
def blas_drot(x: 'float64[:]', y: 'float64[:]', c: 'float64', s: 'float64',
              incx: 'int32' = 1,
              incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.drot (n, x, incx, y, incy, c, s)

# ==============================================================================
def blas_drotm(x: 'float64[:]', y: 'float64[:]', param: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.drotm (n, x, incx, y, incy, param)

# ==============================================================================
def blas_dcopy(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.dcopy (n, x, incx, y, incy)

# ==============================================================================
def blas_dswap(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.dswap (n, x, incx, y, incy)

# ==============================================================================
def blas_dscal(alpha: 'float64', x: 'float64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.dscal (n, alpha, x, incx)

# ==============================================================================
def blas_ddot(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.ddot (n, x, incx, y, incy)

# ==============================================================================
def blas_daxpy(x: 'float64[:]', y: 'float64[:]',
               a: 'float64' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.daxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_dnrm2(x: 'float64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.dnrm2 (n, x, incx)

# ==============================================================================
def blas_dasum(x: 'float64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.dasum (n, x, incx)

# ==============================================================================
def blas_idamax(x: 'float64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    i = mod_blas.idamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def blas_dgemv(alpha: 'float64', a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.dgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dgbmv(kl : 'int32', ku: 'int32', alpha: 'float64',
               a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])

    lda = m
#    lda = np.int32(1) + ku + kl

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.dgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dsymv(alpha: 'float64', a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dsymv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dsbmv(k : 'int32', alpha: 'float64',
               a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dsbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dspmv(alpha: 'float64', a: 'float64[:]', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dspmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_dtrmv(a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.dtrmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_dtbmv(k : 'int32', a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.dtbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_dtpmv(a: 'float64[:]', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.dtpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_dtrsv(a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.dtrsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_dtbsv(k: 'int32', a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.dtbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_dtpsv(a: 'float64[:]', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.dtpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_dger(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.dger (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_dsyr(alpha: 'float64', x: 'float64[:]', a: 'float64[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dsyr (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_dspr(alpha: 'float64', x: 'float64[:]', a: 'float64[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dspr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_dsyr2(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dsyr2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_dspr2(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dspr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def blas_dgemm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               trans_a: 'bool' = False,
               trans_b: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    l = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'

    flag_trans_b = 'N'
    if trans_b: flag_trans_b = 'T'
    # ...

    # ...
    if trans_a:
        m = np.int32(a.shape[0])
    else:
        m = np.int32(a.shape[1])
    # ...

    # TODO to be checked
    lda = m
    ldb = m
    ldc = l

    mod_blas.dgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_dsymm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldb = m
    ldc = m

    mod_blas.dsymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_dsyrk(alpha: 'float64', a: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])
    k = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_trans = 'N'
    lda = n
    # equation 2
    if trans:
        flag_trans = 'T'
        lda = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.dsyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_dsyr2k(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])

    # ...
    # equation 1
    flag_trans = 'N'
    k = np.int32(a.shape[1])
    lda = n
    ldb = n
    # equation 2
    if trans:
        flag_trans = 'T'
        k = np.int32(a.shape[0])
        lda = k
        ldb = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.dsyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_dtrmm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    mod_blas.dtrmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_dtrsm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    mod_blas.dtrsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def blas_ccopy(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.ccopy (n, x, incx, y, incy)

# ==============================================================================
def blas_cswap(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.cswap (n, x, incx, y, incy)

# ==============================================================================
def blas_cscal(alpha: 'complex64', x: 'complex64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.cscal (n, alpha, x, incx)

# ==============================================================================
def blas_cdotc(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.cdotc (n, x, incx, y, incy)

# ==============================================================================
def blas_cdotu(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.cdotu (n, x, incx, y, incy)

# ==============================================================================
def blas_caxpy(x: 'complex64[:]', y: 'complex64[:]',
               a: 'complex64' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.caxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_scnrm2(x: 'complex64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.scnrm2 (n, x, incx)

# ==============================================================================
def blas_scasum(x: 'complex64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.scasum (n, x, incx)

# ==============================================================================
def blas_icamax(x: 'complex64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    i = mod_blas.icamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def blas_cgemv(alpha: 'complex64', a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.cgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_cgbmv(kl : 'int32', ku: 'int32', alpha: 'complex64',
               a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])

    lda = m
#    lda = np.int32(1) + ku + kl

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.cgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_chemv(alpha: 'complex64', a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chemv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_chbmv(k : 'int32', alpha: 'complex64',
               a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_chpmv(alpha: 'complex64', a: 'complex64[:]', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chpmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_ctrmv(a: 'complex64[:,:](order=F)', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ctrmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ctbmv(k : 'int32', a: 'complex64[:,:](order=F)', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ctbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ctpmv(a: 'complex64[:]', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ctpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_ctrsv(a: 'complex64[:,:](order=F)', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ctrsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ctbsv(k: 'int32', a: 'complex64[:,:](order=F)', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ctbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ctpsv(a: 'complex64[:]', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ctpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_cgeru(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.cgeru (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_cgerc(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.cgerc (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_cher(alpha: 'float32', x: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.cher (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_chpr(alpha: 'float32', x: 'complex64[:]', a: 'complex64[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chpr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_cher2(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.cher2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_chpr2(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chpr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def blas_cgemm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               trans_a: 'bool' = False,
               trans_b: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    l = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'

    flag_trans_b = 'N'
    if trans_b: flag_trans_b = 'T'
    # ...

    # ...
    if trans_a:
        m = np.int32(a.shape[0])
    else:
        m = np.int32(a.shape[1])
    # ...

    # TODO to be checked
    lda = m
    ldb = m
    ldc = l

    mod_blas.cgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_csymm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldb = m
    ldc = m

    mod_blas.csymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_chemm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldb = m
    ldc = m

    mod_blas.chemm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_csyrk(alpha: 'complex64', a: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])
    k = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_trans = 'N'
    lda = n
    # equation 2
    if trans:
        flag_trans = 'T'
        lda = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.csyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_csyr2k(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])

    # ...
    # equation 1
    flag_trans = 'N'
    k = np.int32(a.shape[1])
    lda = n
    ldb = n
    # equation 2
    if trans:
        flag_trans = 'T'
        k = np.int32(a.shape[0])
        lda = k
        ldb = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.csyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_cherk(alpha: 'complex64', a: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])
    k = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_trans = 'N'
    lda = n
    # equation 2
    if trans:
        flag_trans = 'T'
        lda = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.cherk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_cher2k(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])

    # ...
    # equation 1
    flag_trans = 'N'
    k = np.int32(a.shape[1])
    lda = n
    ldb = n
    # equation 2
    if trans:
        flag_trans = 'T'
        k = np.int32(a.shape[0])
        lda = k
        ldb = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.cher2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ctrmm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    mod_blas.ctrmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_ctrsm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    mod_blas.ctrsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def blas_zcopy(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.zcopy (n, x, incx, y, incy)

# ==============================================================================
def blas_zswap(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.zswap (n, x, incx, y, incy)

# ==============================================================================
def blas_zscal(alpha: 'complex128', x: 'complex128[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.zscal (n, alpha, x, incx)

# ==============================================================================
def blas_zdotc(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.zdotc (n, x, incx, y, incy)

# ==============================================================================
def blas_zdotu(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.zdotu (n, x, incx, y, incy)

# ==============================================================================
def blas_zaxpy(x: 'complex128[:]', y: 'complex128[:]',
               a: 'complex128' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.zaxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_dznrm2(x: 'complex128[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.dznrm2 (n, x, incx)

# ==============================================================================
def blas_dzasum(x: 'complex128[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.dzasum (n, x, incx)

# ==============================================================================
def blas_izamax(x: 'complex128[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    i = mod_blas.izamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def blas_zgemv(alpha: 'complex128', a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.zgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zgbmv(kl : 'int32', ku: 'int32', alpha: 'complex128',
               a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])

    lda = m
#    lda = np.int32(1) + ku + kl

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.zgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zhemv(alpha: 'complex128', a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zhemv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zhbmv(k : 'int32', alpha: 'complex128',
               a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zhbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zhpmv(alpha: 'complex128', a: 'complex128[:]', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zhpmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_ztrmv(a: 'complex128[:,:](order=F)', x: 'complex128[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ztrmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ztbmv(k : 'int32', a: 'complex128[:,:](order=F)', x: 'complex128[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ztbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ztpmv(a: 'complex128[:]', x: 'complex128[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ztpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_ztrsv(a: 'complex128[:,:](order=F)', x: 'complex128[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ztrsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ztbsv(k: 'int32', a: 'complex128[:,:](order=F)', x: 'complex128[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ztbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ztpsv(a: 'complex128[:]', x: 'complex128[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    mod_blas.ztpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_zgeru(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.zgeru (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_zgerc(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.zgerc (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_zher(alpha: 'float64', x: 'complex128[:]', a: 'complex128[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zher (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_zhpr(alpha: 'float64', x: 'complex128[:]', a: 'complex128[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zhpr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_zher2(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zher2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_zhpr2(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zhpr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def blas_zgemm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
               trans_a: 'bool' = False,
               trans_b: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    l = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'

    flag_trans_b = 'N'
    if trans_b: flag_trans_b = 'T'
    # ...

    # ...
    if trans_a:
        m = np.int32(a.shape[0])
    else:
        m = np.int32(a.shape[1])
    # ...

    # TODO to be checked
    lda = m
    ldb = m
    ldc = l

    mod_blas.zgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zsymm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldb = m
    ldc = m

    mod_blas.zsymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zhemm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldb = m
    ldc = m

    mod_blas.zhemm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zsyrk(alpha: 'complex128', a: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])
    k = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_trans = 'N'
    lda = n
    # equation 2
    if trans:
        flag_trans = 'T'
        lda = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.zsyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_zsyr2k(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])

    # ...
    # equation 1
    flag_trans = 'N'
    k = np.int32(a.shape[1])
    lda = n
    ldb = n
    # equation 2
    if trans:
        flag_trans = 'T'
        k = np.int32(a.shape[0])
        lda = k
        ldb = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.zsyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zherk(alpha: 'complex128', a: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])
    k = np.int32(c.shape[1])

    # ...
    # equation 1
    flag_trans = 'N'
    lda = n
    # equation 2
    if trans:
        flag_trans = 'T'
        lda = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.zherk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_zher2k(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(c.shape[0])

    # ...
    # equation 1
    flag_trans = 'N'
    k = np.int32(a.shape[1])
    lda = n
    ldb = n
    # equation 2
    if trans:
        flag_trans = 'T'
        k = np.int32(a.shape[0])
        lda = k
        ldb = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    mod_blas.zher2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ztrmm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    mod_blas.ztrmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_ztrsm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    mod_blas.ztrsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)
