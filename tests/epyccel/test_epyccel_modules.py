# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np
from pyccel.epyccel import epyccel

def test_module_1():
    import modules.Module_1 as mod

    modnew = epyccel(mod)

    from numpy import zeros

    # ...
    x_expected = zeros(5)
    x          = zeros(5)

    mod.f(x_expected)
    mod.g(x_expected)

    modnew.f(x)
    modnew.g(x)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

def test_local_module_1():
    import Module_1 as mod

    modnew = epyccel(mod)

    from numpy import zeros

    # ...
    x_expected = zeros(5)
    x          = zeros(5)

    mod.f(x_expected)
    mod.g(x_expected)

    modnew.f(x)
    modnew.g(x)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

def test_module_2():
    import modules.Module_2 as mod

    modnew = epyccel(mod)

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2))
    modnew.f6(m1, m2, x)

    x_expected = np.zeros((m1,m2))
    mod.f6(m1, m2, x_expected)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

def test_module_3(language):
    import modules.call_user_defined_funcs as mod

    modnew = epyccel(mod, language=language)

    r = 4.5
    x_expected = mod.circle_volume(r)
    x = modnew.circle_volume(r)
    assert np.isclose( x, x_expected, rtol=1e-14, atol=1e-14 )
