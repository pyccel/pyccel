# pylint: disable=missing-function-docstring, missing-module-docstring/
import sys
import pytest
import numpy as np
import scipy.linalg.blas as sp_blas
import modules.blas as mod
from pyccel.epyccel import epyccel

WIN32_ERROR = "Compilation problem. On execution Windows raises: error while loading shared libraries: libblas.dll: cannot open shared object file: No such file or directory"

# ==============================================================================
def symmetrize(a, lower=False):
    n = a.shape[0]
    if lower:
        for j in range(n):
            for i in range(j):
                a[i,j] = a[j,i]
    else:
        for i in range(n):
            for j in range(i):
                a[i,j] = a[j,i]

    return a

# ==============================================================================
def triangulize(a, lower=False):
    n = a.shape[0]
    if lower:
        for j in range(n):
            for i in range(j):
                a[i,j] = 0.
    else:
        for i in range(n):
            for j in range(i):
                a[i,j] = 0.

    return a

# ==============================================================================
def general_to_band(kl, ku, a):
    n = a.shape[1]
    ab = np.zeros((kl+ku+1, n), dtype=a.dtype)

    for j in range(n):
        k = ku - j
        i1 = max (j - ku, 0)
        i2 = min (j + kl + 1, n)
        for i in range(i1, i2):
            ab[k+i,j] = a[i,j]

    return ab

# ==============================================================================
def general_to_packed(a, lower=False):
    n = a.shape[0]
    ap = np.zeros(n*(n+1)//2, dtype=a.dtype)
    if lower:
        k = 0
        for j in range(n):
            for i in range(j,n):
                ap[k] = a[i,j]
                k += 1
    else:
        k = 0
        for j in range(n):
            for i in range(j+1):
                ap[k] = a[i,j]
                k += 1

    return ap

# ==============================================================================
def random_array(n, dtype):
    np.random.seed(2021)

    if dtype in [np.complex64, np.complex128]:
        x = np.random.random(n) + np.random.random(n) * 1j
    else:
        x = np.random.random(n)

    if len(x.shape) > 1:
        x = x.copy(order='F')

    return np.array(x, dtype=dtype)

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_srotg_1():
    blas_srotg = epyccel( mod.blas_srotg, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    a = b = np.float32(1.)
    c, s = blas_srotg (a, b)
    expected_c, expected_s = sp_blas.srotg (a, b)
    assert(np.abs(c - expected_c) < 1.e-10)
    assert(np.abs(s - expected_s) < 1.e-10)

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_srotmg_1():
    blas_srotmg = epyccel( mod.blas_srotmg, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    d1 = d2 = np.float32(1.)
    x1 = y1 = np.float32(.5)
    result = np.zeros(5, dtype=np.float32)
    blas_srotmg (d1, d2, x1, y1, result)
    expected = sp_blas.srotmg (d1, d2, x1, y1)
    assert(np.allclose(result, expected, TOL))

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_srot_1():
    blas_srot = epyccel( mod.blas_srot, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    one = np.float32(1.)
    c, s = sp_blas.srotg (one, one)
    c = np.float32(c)
    s = np.float32(s)
    expected_x, expected_y = sp_blas.srot(x, y, c, s)
    blas_srot (x, y, c, s)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_srotm_1():
    blas_srotm = epyccel( mod.blas_srotm, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)
    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    d1 = d2 = np.float32(1.)
    x1 = y1 = np.float32(.5)
    param = sp_blas.srotmg (d1, d2, x1, y1)
    param = np.array(param, dtype=np.float32)
    expected_x, expected_y = sp_blas.srotm(x, y, param)
    blas_srotm (x, y, param)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_scopy_1():
    blas_scopy = epyccel( mod.blas_scopy, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected  = np.zeros(n, dtype=np.float32)
    sp_blas.scopy(x, expected)
    blas_scopy (x, y)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sswap_1():
    blas_sswap = epyccel( mod.blas_sswap, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.sswap (x, y)
    blas_sswap (x, y)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sscal_1():
    blas_sscal = epyccel( mod.blas_sscal, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = np.float32(2.5)
    sp_blas.sscal (alpha, x)
    blas_sscal (np.float32(1./alpha), x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sdot_1():
    blas_sdot = epyccel( mod.blas_sdot, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.sdot(x, y)
    result   = blas_sdot (x, y)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sdsdot_1():
    blas_sdsdot = epyccel( mod.blas_sdsdot, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    sb = np.float32(3.)
    # NOTE sdsdot is not implemented in scipy
    expected = sb + sp_blas.sdot(x, y)
    result   = blas_sdsdot (sb, x, y)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dsdot_1():
    blas_dsdot = epyccel( mod.blas_dsdot, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    # NOTE dsdot is not implemented in scipy
    expected = sp_blas.sdot(x, y)
    result   = blas_dsdot (x, y)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_snrm2_1():
    blas_snrm2 = epyccel( mod.blas_snrm2, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.snrm2(x)
    result   = blas_snrm2 (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sasum_1():
    blas_sasum = epyccel( mod.blas_sasum, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.sasum(x)
    result   = blas_sasum (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_isamax_1():
    blas_isamax = epyccel( mod.blas_isamax, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.isamax(x)
    result   = blas_isamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_saxpy_1():
    blas_saxpy = epyccel( mod.blas_saxpy, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(2.5)
    expected = y.copy()
    sp_blas.saxpy (x, expected, a=alpha)
    blas_saxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sgemv_1():
    blas_sgemv = epyccel( mod.blas_sgemv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = y.copy()
    expected = sp_blas.sgemv (alpha, a, x, beta=beta, y=expected)
    blas_sgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sgbmv_1():
    blas_sgbmv = epyccel( mod.blas_sgbmv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 5
    kl = np.int32(2)
    ku = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.float32)

    ab = general_to_band(kl, ku, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = y.copy()
    expected = sp_blas.sgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_sgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ssymv_1():
    blas_ssymv = epyccel( mod.blas_ssymv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = sp_blas.ssymv (alpha, a, x, y=y, beta=beta)
    blas_ssymv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ssbmv_1():
    blas_ssbmv = epyccel( mod.blas_ssbmv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.float32)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = y.copy()
    expected = sp_blas.ssbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_ssbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sspmv_1():
    blas_sspmv = epyccel( mod.blas_sspmv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = sp_blas.sspmv (n, alpha, ap, x, y=y, beta=beta)
    blas_sspmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_strmv_1():
    blas_strmv = epyccel( mod.blas_strmv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.strmv (a, x)
    blas_strmv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_stbmv_1():
    blas_stbmv = epyccel( mod.blas_stbmv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.float32)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.stbmv (k, ab, x)
    blas_stbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_stpmv_1():
    blas_stpmv = epyccel( mod.blas_stpmv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    expected = sp_blas.stpmv (n, ap, x)
    blas_stpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_strsv_1():
    blas_strsv = epyccel( mod.blas_strsv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    b = x.copy()
    expected = sp_blas.strsv (a, b)
    blas_strsv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_stbsv_1():
    blas_stbsv = epyccel( mod.blas_stbsv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.float32)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.stbsv (k, ab, x)
    blas_stbsv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_stpsv_1():
    blas_stpsv = epyccel( mod.blas_stpsv, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    b = x.copy()
    expected = sp_blas.stpsv (n, ap, b)
    blas_stpsv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sger_1():
    blas_sger = epyccel( mod.blas_sger, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.sger (alpha, x, y, a=a)
    blas_sger (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ssyr_1():
    blas_ssyr = epyccel( mod.blas_ssyr, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.ssyr (alpha, x, a=a)
    blas_ssyr (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sspr_1():
    blas_sspr = epyccel( mod.blas_sspr, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.sspr (n, alpha, x, ap)
    blas_sspr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ssyr2_1():
    blas_ssyr2 = epyccel( mod.blas_ssyr2, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.ssyr2 (alpha, x, y, a=a)
    blas_ssyr2 (alpha, x, y, a=a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sspr2_1():
    blas_sspr2 = epyccel( mod.blas_sspr2, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.sspr2 (n, alpha, x, y, ap)
    blas_sspr2 (alpha, x, y, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_sgemm_1():
    blas_sgemm = epyccel( mod.blas_sgemm, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = sp_blas.sgemm (alpha, a, b, c=c, beta=beta)
    blas_sgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ssymm_1():
    blas_ssymm = epyccel( mod.blas_ssymm, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = sp_blas.ssymm (alpha, a, b, c=c, beta=beta)
    blas_ssymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_strmm_1():
    blas_strmm = epyccel( mod.blas_strmm, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.strmm (alpha, a, b)
    blas_strmm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_strsm_1():
    blas_strsm = epyccel( mod.blas_strsm, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.strsm (alpha, a, b)
    blas_strsm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ssyrk_1():
    blas_ssyrk = epyccel( mod.blas_ssyrk, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.)
    expected = sp_blas.ssyrk (alpha, a, c=c, beta=beta)
    blas_ssyrk (alpha, a, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ssyr2k_1():
    blas_ssyr2k = epyccel( mod.blas_ssyr2k, language = 'fortran' )

    TOL = 1.e-7
    DTYPE = np.float32

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.)
    expected = sp_blas.ssyr2k (alpha, a, b, c=c, beta=beta)
    blas_ssyr2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_drotg_1():
    blas_drotg = epyccel( mod.blas_drotg, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    a = b = np.float64(1.)
    c, s = blas_drotg (a, b)
    expected_c, expected_s = sp_blas.drotg (a, b)
    assert(np.abs(c - expected_c) < 1.e-10)
    assert(np.abs(s - expected_s) < 1.e-10)

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_drotmg_1():
    blas_drotmg = epyccel( mod.blas_drotmg, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    d1 = d2 = np.float64(1.)
    x1 = y1 = np.float64(.5)
    result = np.zeros(5, dtype=np.float64)
    blas_drotmg (d1, d2, x1, y1, result)
    expected = sp_blas.drotmg (d1, d2, x1, y1)
    assert(np.allclose(result, expected, TOL))

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_drot_1():
    blas_drot = epyccel( mod.blas_drot, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    one = np.float64(1.)
    c, s = sp_blas.drotg (one, one)
    c = np.float64(c)
    s = np.float64(s)
    expected_x, expected_y = sp_blas.drot(x, y, c, s)
    blas_drot (x, y, c, s)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_drotm_1():
    blas_drotm = epyccel( mod.blas_drotm, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    d1 = d2 = np.float64(1.)
    x1 = y1 = np.float64(.5)
    param = sp_blas.drotmg (d1, d2, x1, y1)
    param = np.array(param, dtype=np.float64)
    expected_x, expected_y = sp_blas.drotm(x, y, param)
    blas_drotm (x, y, param)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dcopy_1():
    blas_dcopy = epyccel( mod.blas_dcopy, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected  = np.zeros(n, dtype=np.float64)
    sp_blas.dcopy(x, expected)
    blas_dcopy (x, y)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dswap_1():
    blas_dswap = epyccel( mod.blas_dswap, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.dswap (x, y)
    blas_dswap (x, y)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dscal_1():
    blas_dscal = epyccel( mod.blas_dscal, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = np.float64(2.5)
    sp_blas.dscal (alpha, x)
    blas_dscal (np.float64(1./alpha), x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ddot_1():
    blas_ddot = epyccel( mod.blas_ddot, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.ddot(x, y)
    result   = blas_ddot (x, y)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dnrm2_1():
    blas_dnrm2 = epyccel( mod.blas_dnrm2, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.dnrm2(x)
    result   = blas_dnrm2 (x)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dasum_1():
    blas_dasum = epyccel( mod.blas_dasum, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.dasum(x)
    result   = blas_dasum (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_idamax_1():
    blas_idamax = epyccel( mod.blas_idamax, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.idamax(x)
    result   = blas_idamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_daxpy_1():
    blas_daxpy = epyccel( mod.blas_daxpy, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float64(2.5)
    expected = y.copy()
    sp_blas.daxpy (x, expected, a=alpha)
    blas_daxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dgemv_1():
    blas_dgemv = epyccel( mod.blas_dgemv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = y.copy()
    expected = sp_blas.dgemv (alpha, a, x, beta=beta, y=expected)
    blas_dgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dgbmv_1():
    blas_dgbmv = epyccel( mod.blas_dgbmv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 5
    kl = np.int32(2)
    ku = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.float64)

    ab = general_to_band(kl, ku, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = y.copy()
    expected = sp_blas.dgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_dgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dsymv_1():
    blas_dsymv = epyccel( mod.blas_dsymv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = sp_blas.dsymv (alpha, a, x, y=y, beta=beta)
    blas_dsymv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dsbmv_1():
    blas_dsbmv = epyccel( mod.blas_dsbmv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.float64)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = y.copy()
    expected = sp_blas.dsbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_dsbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dspmv_1():
    blas_dspmv = epyccel( mod.blas_dspmv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = sp_blas.dspmv (n, alpha, ap, x, y=y, beta=beta)
    blas_dspmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dtrmv_1():
    blas_dtrmv = epyccel( mod.blas_dtrmv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.dtrmv (a, x)
    blas_dtrmv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dtbmv_1():
    blas_dtbmv = epyccel( mod.blas_dtbmv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.float64)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.dtbmv (k, ab, x)
    blas_dtbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dtpmv_1():
    blas_dtpmv = epyccel( mod.blas_dtpmv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    expected = sp_blas.dtpmv (n, ap, x)
    blas_dtpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dtrsv_1():
    blas_dtrsv = epyccel( mod.blas_dtrsv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    b = x.copy()
    expected = sp_blas.dtrsv (a, b)
    blas_dtrsv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dtbsv_1():
    blas_dtbsv = epyccel( mod.blas_dtbsv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.float64)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.dtbsv (k, ab, x)
    blas_dtbsv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dtpsv_1():
    blas_dtpsv = epyccel( mod.blas_dtpsv, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    b = x.copy()
    expected = sp_blas.dtpsv (n, ap, b)
    blas_dtpsv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dger_1():
    blas_dger = epyccel( mod.blas_dger, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dger (alpha, x, y, a=a)
    blas_dger (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dsyr_1():
    blas_dsyr = epyccel( mod.blas_dsyr, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dsyr (alpha, x, a=a)
    blas_dsyr (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dspr_1():
    blas_dspr = epyccel( mod.blas_dspr, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dspr (n, alpha, x, ap)
    blas_dspr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dsyr2_1():
    blas_dsyr2 = epyccel( mod.blas_dsyr2, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dsyr2 (alpha, x, y, a=a)
    blas_dsyr2 (alpha, x, y, a=a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dspr2_1():
    blas_dspr2 = epyccel( mod.blas_dspr2, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dspr2 (n, alpha, x, y, ap)
    blas_dspr2 (alpha, x, y, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dgemm_1():
    blas_dgemm = epyccel( mod.blas_dgemm, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = sp_blas.dgemm (alpha, a, b, c=c, beta=beta)
    blas_dgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dsymm_1():
    blas_dsymm = epyccel( mod.blas_dsymm, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = sp_blas.dsymm (alpha, a, b, c=c, beta=beta)
    blas_dsymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dtrmm_1():
    blas_dtrmm = epyccel( mod.blas_dtrmm, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dtrmm (alpha, a, b)
    blas_dtrmm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dtrsm_1():
    blas_dtrsm = epyccel( mod.blas_dtrsm, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dtrsm (alpha, a, b)
    blas_dtrsm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dsyrk_1():
    blas_dsyrk = epyccel( mod.blas_dsyrk, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.)
    expected = sp_blas.dsyrk (alpha, a, c=c, beta=beta)
    blas_dsyrk (alpha, a, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dsyr2k_1():
    blas_dsyr2k = epyccel( mod.blas_dsyr2k, language = 'fortran' )

    TOL = 1.e-13
    DTYPE = np.float64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.)
    expected = sp_blas.dsyr2k (alpha, a, b, c=c, beta=beta)
    blas_dsyr2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ccopy_1():
    blas_ccopy = epyccel( mod.blas_ccopy, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 3
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = y.copy()
    sp_blas.ccopy(x, expected)
    blas_ccopy (x, y)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cswap_1():
    blas_cswap = epyccel( mod.blas_cswap, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.cswap (x, y)
    blas_cswap (x, y)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cscal_1():
    blas_cscal = epyccel( mod.blas_cscal, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = np.complex64(2.5)
    inv_alpha = np.complex64(1./alpha)
    sp_blas.cscal (alpha, x)
    blas_cscal (inv_alpha, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_scnrm2_1():
    blas_scnrm2 = epyccel( mod.blas_scnrm2, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.scnrm2(x)
    result   = blas_scnrm2 (x)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_scasum_1():
    blas_scasum = epyccel( mod.blas_scasum, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.scasum(x)
    result   = blas_scasum (x)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_icamax_1():
    blas_icamax = epyccel( mod.blas_icamax, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.icamax(x)
    result   = blas_icamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_caxpy_1():
    blas_caxpy = epyccel( mod.blas_caxpy, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex64(2.5)
    expected = y.copy()
    sp_blas.caxpy (x, expected, a=alpha)
    blas_caxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cdotc_1():
    blas_cdotc = epyccel( mod.blas_cdotc, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 3
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.cdotc(x, y)
    result   = blas_cdotc (x, y)
    assert(np.linalg.norm(result-expected) < 1.e-6)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cdotu_1():
    blas_cdotu = epyccel( mod.blas_cdotu, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.cdotu(x, y)
    result   = blas_cdotu (x, y)
    assert(np.linalg.norm(result-expected) < 1.e-6)
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cgemv_1():
    blas_cgemv = epyccel( mod.blas_cgemv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex64(1.0)
    beta = np.complex64(0.5)
    expected = y.copy()
    expected = sp_blas.cgemv (alpha, a, x, beta=beta, y=expected)
    blas_cgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cgbmv_1():
    blas_cgbmv = epyccel( mod.blas_cgbmv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 5
    kl = np.int32(2)
    ku = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.complex64)

    ab = general_to_band(kl, ku, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex64(1.0)
    beta = np.complex64(0.5)
    expected = y.copy()
    expected = sp_blas.cgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_cgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_chemv_1():
    blas_chemv = epyccel( mod.blas_chemv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex64(1.0)
    beta = np.complex64(0.5)
    expected = y.copy()
    expected = sp_blas.chemv (alpha, a, x, beta=beta, y=expected)
    blas_chemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_chbmv_1():
    blas_chbmv = epyccel( mod.blas_chbmv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.complex64)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex64(1.0)
    beta = np.complex64(0.5)
    expected = y.copy()
    expected = sp_blas.chbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_chbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_chpmv_1():
    blas_chpmv = epyccel( mod.blas_chpmv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(0.5)
    expected = sp_blas.chpmv (n, alpha, ap, x, y=y, beta=beta)
    blas_chpmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ctrmv_1():
    blas_ctrmv = epyccel( mod.blas_ctrmv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.ctrmv (a, x)
    blas_ctrmv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ctbmv_1():
    blas_ctbmv = epyccel( mod.blas_ctbmv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.complex64)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.ctbmv (k, ab, x)
    blas_ctbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ctpmv_1():
    blas_ctpmv = epyccel( mod.blas_ctpmv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    expected = sp_blas.ctpmv (n, ap, x)
    blas_ctpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ctrsv_1():
    blas_ctrsv = epyccel( mod.blas_ctrsv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    b = x.copy()
    expected = sp_blas.ctrsv (a, b)
    blas_ctrsv (a, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ctbsv_1():
    blas_ctbsv = epyccel( mod.blas_ctbsv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.complex64)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.ctbsv (k, ab, x)
    blas_ctbsv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ctpsv_1():
    blas_ctpsv = epyccel( mod.blas_ctpsv, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    b = x.copy()
    expected = sp_blas.ctpsv (n, ap, b)
    blas_ctpsv (ap, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cgeru_1():
    blas_cgeru = epyccel( mod.blas_cgeru, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.cgeru (alpha, x, y, a=a)
    blas_cgeru (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cgerc_1():
    blas_cgerc = epyccel( mod.blas_cgerc, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.cgerc (alpha, x, y, a=a)
    blas_cgerc (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cher_1():
    blas_cher = epyccel( mod.blas_cher, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.cher (alpha, x, a=a)
    blas_cher (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_chpr_1():
    blas_chpr = epyccel( mod.blas_chpr, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.chpr (n, alpha, x, ap)
    blas_chpr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cher2_1():
    blas_cher2 = epyccel( mod.blas_cher2, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.cher2 (alpha, x, y, a=a)
    blas_cher2 (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_chpr2_1():
    blas_chpr2 = epyccel( mod.blas_chpr2, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.chpr2 (n, alpha, x, y, ap)
    blas_chpr2 (alpha, x, y, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cgemm_1():
    blas_cgemm = epyccel( mod.blas_cgemm, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.cgemm (alpha, a, b, c=c, beta=beta)
    blas_cgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_csymm_1():
    blas_csymm = epyccel( mod.blas_csymm, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.csymm (alpha, a, b, c=c, beta=beta)
    blas_csymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_chemm_1():
    blas_chemm = epyccel( mod.blas_chemm, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.chemm (alpha, a, b, beta=beta, c=c)
    blas_chemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_csyrk_1():
    blas_csyrk = epyccel( mod.blas_csyrk, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.csyrk (alpha, a, beta=beta, c=c)
    blas_csyrk (alpha, a, c, beta=beta)
    assert(np.linalg.norm(c- expected) < TOL)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_csyr2k_1():
    blas_csyr2k = epyccel( mod.blas_csyr2k, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.csyr2k (alpha, a, b, beta=beta, c=c)
    blas_csyr2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cherk_1():
    blas_cherk = epyccel( mod.blas_cherk, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.cherk (alpha, a, beta=beta, c=c)
    blas_cherk (alpha, a, c, beta=beta)
    assert(np.linalg.norm(c- expected) < TOL)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_cher2k_1():
    blas_cher2k = epyccel( mod.blas_cher2k, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.cher2k (alpha, a, b, beta=beta, c=c)
    blas_cher2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ctrmm_1():
    blas_ctrmm = epyccel( mod.blas_ctrmm, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.ctrmm (alpha, a, b)
    blas_ctrmm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ctrsm_1():
    blas_ctrsm = epyccel( mod.blas_ctrsm, language = 'fortran' )

    TOL = 1.e-6
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.ctrsm (alpha, a, b)
    blas_ctrsm (alpha, a, b)
    assert(np.linalg.norm(b- expected) < TOL)
    # ...

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zcopy_1():
    blas_zcopy = epyccel( mod.blas_zcopy, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 3
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = y.copy()
    sp_blas.zcopy(x, expected)
    blas_zcopy (x, y)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zswap_1():
    blas_zswap = epyccel( mod.blas_zswap, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.zswap (x, y)
    blas_zswap (x, y)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zscal_1():
    blas_zscal = epyccel( mod.blas_zscal, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = np.complex128(2.5)
    inv_alpha = np.complex128(1./alpha)
    sp_blas.zscal (alpha, x)
    blas_zscal (inv_alpha, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dznrm2_1():
    blas_dznrm2 = epyccel( mod.blas_dznrm2, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.dznrm2(x)
    result   = blas_dznrm2 (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_dzasum_1():
    blas_dzasum = epyccel( mod.blas_dzasum, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.dzasum(x)
    result   = blas_dzasum (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_izamax_1():
    blas_izamax = epyccel( mod.blas_izamax, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.izamax(x)
    result   = blas_izamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zaxpy_1():
    blas_zaxpy = epyccel( mod.blas_zaxpy, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex128(2.5)
    expected = y.copy()
    sp_blas.zaxpy (x, expected, a=alpha)
    blas_zaxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zdotc_1():
    blas_zdotc = epyccel( mod.blas_zdotc, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 3
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.zdotc(x, y)
    result   = blas_zdotc (x, y)
    assert(np.linalg.norm(result-expected) < TOL)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zdotu_1():
    blas_zdotu = epyccel( mod.blas_zdotu, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.zdotu(x, y)
    result   = blas_zdotu (x, y)
    assert(np.linalg.norm(result-expected) < TOL)
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zgemv_1():
    blas_zgemv = epyccel( mod.blas_zgemv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex128(1.0)
    beta = np.complex128(0.5)
    expected = y.copy()
    expected = sp_blas.zgemv (alpha, a, x, beta=beta, y=expected)
    blas_zgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zgbmv_1():
    blas_zgbmv = epyccel( mod.blas_zgbmv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 5
    kl = np.int32(2)
    ku = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.complex128)

    ab = general_to_band(kl, ku, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex128(1.0)
    beta = np.complex128(0.5)
    expected = y.copy()
    expected = sp_blas.zgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_zgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zhemv_1():
    blas_zhemv = epyccel( mod.blas_zhemv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex128(1.0)
    beta = np.complex128(0.5)
    expected = y.copy()
    expected = sp_blas.zhemv (alpha, a, x, beta=beta, y=expected)
    blas_zhemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zhbmv_1():
    blas_zhbmv = epyccel( mod.blas_zhbmv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.complex128)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex128(1.0)
    beta = np.complex128(0.5)
    expected = y.copy()
    expected = sp_blas.zhbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_zhbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zhpmv_1():
    blas_zhpmv = epyccel( mod.blas_zhpmv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(0.5)
    expected = sp_blas.zhpmv (n, alpha, ap, x, y=y, beta=beta)
    blas_zhpmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ztrmv_1():
    blas_ztrmv = epyccel( mod.blas_ztrmv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.ztrmv (a, x)
    blas_ztrmv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ztbmv_1():
    blas_ztbmv = epyccel( mod.blas_ztbmv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.complex128)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.ztbmv (k, ab, x)
    blas_ztbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ztpmv_1():
    blas_ztpmv = epyccel( mod.blas_ztpmv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    expected = sp_blas.ztpmv (n, ap, x)
    blas_ztpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ztrsv_1():
    blas_ztrsv = epyccel( mod.blas_ztrsv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    b = x.copy()
    expected = sp_blas.ztrsv (a, b)
    blas_ztrsv (a, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ztbsv_1():
    blas_ztbsv = epyccel( mod.blas_ztbsv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.complex128)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.ztbsv (k, ab, x)
    blas_ztbsv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ztpsv_1():
    blas_ztpsv = epyccel( mod.blas_ztpsv, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    x.copy()
    expected = sp_blas.ztpsv (n, ap, x)
    blas_ztpsv (ap, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zgeru_1():
    blas_zgeru = epyccel( mod.blas_zgeru, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zgeru (alpha, x, y, a=a)
    blas_zgeru (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zgerc_1():
    blas_zgerc = epyccel( mod.blas_zgerc, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zgerc (alpha, x, y, a=a)
    blas_zgerc (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zher_1():
    blas_zher = epyccel( mod.blas_zher, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.zher (alpha, x, a=a)
    blas_zher (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zhpr_1():
    blas_zhpr = epyccel( mod.blas_zhpr, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.zhpr (n, alpha, x, ap)
    blas_zhpr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zher2_1():
    blas_zher2 = epyccel( mod.blas_zher2, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zher2 (alpha, x, y, a=a)
    blas_zher2 (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zhpr2_1():
    blas_zhpr2 = epyccel( mod.blas_zhpr2, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zhpr2 (n, alpha, x, y, ap)
    blas_zhpr2 (alpha, x, y, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zgemm_1():
    blas_zgemm = epyccel( mod.blas_zgemm, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zgemm (alpha, a, b, c=c, beta=beta)
    blas_zgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zsymm_1():
    blas_zsymm = epyccel( mod.blas_zsymm, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zsymm (alpha, a, b, c=c, beta=beta)
    blas_zsymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zhemm_1():
    blas_zhemm = epyccel( mod.blas_zhemm, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zhemm (alpha, a, b, beta=beta, c=c)
    blas_zhemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zsyrk_1():
    blas_zsyrk = epyccel( mod.blas_zsyrk, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zsyrk (alpha, a, beta=beta, c=c)
    blas_zsyrk (alpha, a, c, beta=beta)
    assert(np.linalg.norm(c- expected) < TOL)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zsyr2k_1():
    blas_zsyr2k = epyccel( mod.blas_zsyr2k, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zsyr2k (alpha, a, b, beta=beta, c=c)
    blas_zsyr2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zherk_1():
    blas_zherk = epyccel( mod.blas_zherk, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zherk (alpha, a, beta=beta, c=c)
    blas_zherk (alpha, a, c, beta=beta)
    assert(np.linalg.norm(c- expected) < TOL)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_zher2k_1():
    blas_zher2k = epyccel( mod.blas_zher2k, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zher2k (alpha, a, b, beta=beta, c=c)
    blas_zher2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ztrmm_1():
    blas_ztrmm = epyccel( mod.blas_ztrmm, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.ztrmm (alpha, a, b)
    blas_ztrmm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason=WIN32_ERROR)
def test_ztrsm_1():
    blas_ztrsm = epyccel( mod.blas_ztrsm, language = 'fortran' )

    TOL = 1.e-12
    DTYPE = np.complex128

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.ztrsm (alpha, a, b)
    blas_ztrsm (alpha, a, b)
    assert(np.linalg.norm(b- expected) < TOL)
    # ...
