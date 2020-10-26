# pylint: disable=missing-function-docstring, missing-module-docstring/
from mpi4py import MPI
import pytest
import numpy as np

from pyccel.epyccel import epyccel

@pytest.fixture(params=[
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = [pytest.mark.c,
        pytest.mark.xfail(message='Arrays not implemented in C')])
    ]
)
def language(request):
    return request.param

#==============================================================================
@pytest.mark.parallel
def test_module_1(language):
    import modules.Module_1 as mod

    modnew = epyccel(mod, comm=MPI.COMM_WORLD, language=language)

    # ...
    x_expected = np.zeros(5)
    x          = np.zeros(5)

    mod.f(x_expected)
    mod.g(x_expected)

    modnew.f(x)
    modnew.g(x)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

#==============================================================================
@pytest.mark.parallel
def test_module_2(language):
    import modules.Module_2 as mod

    modnew = epyccel(mod, comm=MPI.COMM_WORLD, language=language)

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2))
    modnew.f6(m1, m2, x)

    x_expected = np.zeros((m1,m2))
    mod.f6(m1, m2, x_expected)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

#==============================================================================
@pytest.mark.parallel
def test_function(language):
    from modules.Module_1 import f, g, h

    comm = MPI.COMM_WORLD

    f_fast = epyccel(f, comm=comm, language=language)
    g_fast = epyccel(g, comm=comm, language=language)
    h_fast = epyccel(h, comm=comm, language=language)

    assert f_fast is not f
    assert g_fast is not g
    assert h_fast is not h

    x_expected = np.zeros(3)
    f(x_expected)
    g(x_expected)
    h(x_expected)

    x = np.zeros(3)
    f_fast(x)
    g_fast(x)
    h_fast(x)

    assert np.array_equal(x, x_expected)
