from mpi4py import MPI
import pytest
import numpy as np

from pyccel.epyccel import epyccel

#==============================================================================
@pytest.mark.parallel
def test_module_1():
    import modules.Module_1 as mod

    modnew = epyccel(mod, comm=MPI.COMM_WORLD)

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
def test_module_2():
    import modules.Module_2 as mod

    modnew = epyccel(mod, comm=MPI.COMM_WORLD)

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2))
    modnew.f6(m1, m2, x)

    x_expected = np.zeros((m1,m2))
    mod.f6(m1, m2, x_expected)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...
