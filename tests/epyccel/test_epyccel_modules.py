# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np
from pyccel.epyccel import epyccel

def test_module_1(language):
    import modules.Module_1 as mod

    modnew = epyccel(mod, language=language)

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

def test_local_module_1(language):
    import Module_1 as mod

    modnew = epyccel(mod, language=language)

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

def test_module_2(language):
    import modules.Module_2 as mod

    modnew = epyccel(mod, language=language)

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

def test_module_4(language):
    import modules.Module_6 as mod

    modnew = epyccel(mod, language=language)

    n_x = np.random.randint(4,20)
    n_y = np.random.randint(4,20)

    x = np.empty(n_x, dtype=float)
    y = np.random.random_sample(n_y)

    x_pyc = x.copy()
    y_pyc = y.copy()

    max_pyt = mod.f(x,y)
    max_pyc = modnew.f(x_pyc, y_pyc)
    assert np.isclose( max_pyt, max_pyc, rtol=1e-14, atol=1e-14 )
    assert np.allclose( x, x_pyc, rtol=1e-14, atol=1e-14 )
    assert np.allclose( y, y_pyc, rtol=1e-14, atol=1e-14 )

def test_module_5(language):
    import modules.Module_7 as mod

    modnew = epyccel(mod, language=language)

    max_pyt = mod.get_sum()
    max_pyc = modnew.get_sum()
    assert np.isclose( max_pyt, max_pyc, rtol=1e-14, atol=1e-14 )

    max_pyt = mod.get_sum2()
    max_pyc = modnew.get_sum2()
    assert np.isclose( max_pyt, max_pyc, rtol=1e-14, atol=1e-14 )
