# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
import pytest
import numpy as np
import scipy.linalg.blas as sp_blas
import modules.external_functions as mod
from pytest_teardown_tools import run_epyccel, clean_test

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason="Compilation problem. On execution Windows raises: error while loading shared libraries: libblas.dll: cannot open shared object file: No such file or directory" )
def test_dnrm2_1():
    blas_dnrm2 = run_epyccel( mod.blas_dnrm2, language = 'fortran' )

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ...
    err_expected = sp_blas.dnrm2(x)
    err_pyccel   = blas_dnrm2(x)
    assert(np.abs(err_pyccel - err_expected) < 1.e-14)
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason="Compilation problem. On execution Windows raises: error while loading shared libraries: libblas.dll: cannot open shared object file: No such file or directory" )
def test_dasum_1():
    blas_dasum = run_epyccel( mod.blas_dasum, language = 'fortran' )

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ...
    expected = sp_blas.dasum(x)
    result   = blas_dasum (x)
    assert(np.allclose(result, expected, 1.e-14))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason="Compilation problem. On execution Windows raises: error while loading shared libraries: libblas.dll: cannot open shared object file: No such file or directory" )
def test_ddot_1():
    blas_ddot = run_epyccel( mod.blas_ddot, language = 'fortran' )

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)

    # ...
    expected = sp_blas.ddot(x, y)
    result   = blas_ddot (x, y)
    assert(np.allclose(result, expected, 1.e-14))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason="Compilation problem. On execution Windows raises: error while loading shared libraries: libblas.dll: cannot open shared object file: No such file or directory" )
def test_ddot_2():
    blas_ddot = run_epyccel( mod.blas_ddot_in_func, language = 'fortran' )

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)

    # ...
    expected = sp_blas.ddot(x, y)
    result   = blas_ddot (x, y)
    assert(np.allclose(result, expected, 1.e-14))
    # ...

# ==============================================================================
@pytest.mark.fortran
@pytest.mark.skipif( sys.platform == 'win32', reason="Compilation problem. On execution Windows raises: error while loading shared libraries: libblas.dll: cannot open shared object file: No such file or directory" )
def test_idamax_1():
    blas_idamax = run_epyccel( mod.blas_idamax, language = 'fortran' )

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ...
    expected = sp_blas.idamax(x)
    result   = blas_idamax (x)
    assert(result == expected)
    # ...

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module(module):
    clean_test()
