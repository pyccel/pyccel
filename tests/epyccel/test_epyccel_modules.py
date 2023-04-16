# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pytest_teardown_tools import run_epyccel, clean_test

RTOL = 2e-14
ATOL = 1e-15

def test_module_1(language):
    import modules.Module_1 as mod

    modnew = run_epyccel(mod, language=language)

    from numpy import zeros

    # ...
    x_expected = zeros(5)
    x          = zeros(5)

    mod.f(x_expected)
    mod.g(x_expected)

    modnew.f(x)
    modnew.g(x)

    assert np.allclose( x, x_expected, rtol=RTOL, atol=ATOL )
    # ...

def test_local_module_1(language):
    import Module_1 as mod

    modnew = run_epyccel(mod, language=language)

    from numpy import zeros

    # ...
    x_expected = zeros(5)
    x          = zeros(5)

    mod.f(x_expected)
    mod.g(x_expected)

    modnew.f(x)
    modnew.g(x)

    assert np.allclose( x, x_expected, rtol=RTOL, atol=ATOL )
    # ...

def test_module_2(language):
    import modules.Module_2 as mod

    modnew = run_epyccel(mod, language=language)

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2))
    modnew.f6(m1, m2, x)

    x_expected = np.zeros((m1,m2))
    mod.f6(m1, m2, x_expected)

    assert np.allclose( x, x_expected, rtol=RTOL, atol=ATOL )
    # ...

def test_module_3(language):
    import modules.call_user_defined_funcs as mod

    modnew = run_epyccel(mod, language=language)

    r = 4.5
    x_expected = mod.circle_volume(r)
    x = modnew.circle_volume(r)
    assert np.isclose( x, x_expected, rtol=RTOL, atol=ATOL )

    i = np.random.randint(4,20)
    n = np.random.randint(2,8)
    arr = np.array(100*np.random.random_sample(n), dtype=int)
    x_expected, y_expected = mod.alias(arr, i)
    x, y = modnew.alias(arr, i)

    assert np.allclose( x, x_expected, rtol=RTOL, atol=ATOL )
    assert np.allclose( y, y_expected, rtol=RTOL, atol=ATOL )
    assert x.dtype is x_expected.dtype
    assert y.dtype is y_expected.dtype

def test_module_4(language):
    import modules.Module_6 as mod

    modnew = run_epyccel(mod, language=language)

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

    modnew = run_epyccel(mod, language=language)

    max_pyt = mod.get_sum()
    max_pyc = modnew.get_sum()
    assert np.isclose( max_pyt, max_pyc, rtol=1e-14, atol=1e-14 )

    max_pyt = mod.get_sum2()
    max_pyc = modnew.get_sum2()
    assert np.isclose( max_pyt, max_pyc, rtol=1e-14, atol=1e-14 )

def test_module_6(language):
    import modules.consts as mod

    modnew = run_epyccel(mod, language=language)

    atts = ('g', 'R0', 'rMin', 'rMax', 'skip_centre',
            'method', 'compl', 'tiny')
    for att in atts:
        mod_att = getattr(mod, att)
        modnew_att = getattr(modnew, att)
        assert mod_att == modnew_att
        assert type(mod_att) is type(modnew_att)

def test_module_7(language):
    import modules.array_consts as mod

    modnew = run_epyccel(mod, language=language)

    atts = ('a', 'b', 'c', 'd', 'e')
    for att in atts:
        mod_att = getattr(mod, att)
        modnew_att = getattr(modnew, att)
        assert np.array_equal(mod_att, modnew_att)
        assert mod_att.dtype == modnew_att.dtype

    assert np.array_equal(mod.F, modnew.F)

    modnew.update_a()
    mod.update_a()

    mod_att = mod.a
    modnew_att = modnew.a
    assert np.array_equal(mod_att, modnew_att)
    assert mod_att.dtype == modnew_att.dtype

    mod.a[3] = 10
    modnew.a[3] = 10
    assert np.array_equal(mod_att, modnew_att)
    assert mod.get_elem_a(3) == modnew.get_elem_a(3)

    mod.c[1,0] = 10
    modnew.c[1,0] = 10
    assert np.array_equal(mod.c, modnew.c)
    assert mod.get_elem_c(1,0) == modnew.get_elem_c(1,0)

    mod.e[1,0,2] = 50
    modnew.e[1,0,2] = 50
    assert np.array_equal(mod.e, modnew.e)
    assert mod.get_elem_e(1,0,2) == modnew.get_elem_e(1,0,2)

    # Necessary as python does not reload modules
    mod.reset_a()
    mod.reset_c()
    mod.reset_e()

def test_awkward_names(language):
    import modules.awkward_names as mod

    modnew = run_epyccel(mod, language=language)

    assert mod.awkward_names == modnew.awkward_names
    assert mod.a == modnew.a
    assert mod.A == modnew.A
    assert mod.function() == modnew.function()
    assert mod.pure() == modnew.pure()
    assert mod.allocate(1) == modnew.allocate(1)

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module(module):
    clean_test()
