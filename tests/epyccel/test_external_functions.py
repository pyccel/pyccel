# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
import numpy as np
import scipy.linalg.blas as sp_blas
import modules.external_functions as mod
from pyccel.epyccel import epyccel

@pytest.fixture(scope="module")
def modnew():
    return epyccel(mod, language = 'fortran')

def test_dnrm2_1(modnew):
    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ...
    err_expected = sp_blas.dnrm2(x)
    err_pyccel   = modnew.blas_dnrm2(x)
    assert(np.abs(err_pyccel - err_expected) < 1.e-14)
    # ...

# ==============================================================================
def test_dasum_1(modnew):
    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ...
    expected = sp_blas.dasum(x)
    result   = modnew.blas_dasum (x)
    assert(np.allclose(result, expected, 1.e-14))
    # ...

# ==============================================================================
def test_ddot_1(modnew):
    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)

    # ...
    expected = sp_blas.ddot(x, y)
    result   = modnew.blas_ddot (x, y)
    assert(np.allclose(result, expected, 1.e-14))
    # ...
# ==============================================================================
def test_idamax_1(modnew):
    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ...
    expected = sp_blas.idamax(x)
    result   = modnew.blas_idamax (x)
    assert(result == expected)
    # ...
