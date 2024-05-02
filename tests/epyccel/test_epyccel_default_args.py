# coding: utf-8

import numpy as np
import shutil

from pyccel.epyccel import epyccel
from pyccel.decorators import types


def clean_test():
    shutil.rmtree('__pycache__', ignore_errors=True)
    shutil.rmtree('__epyccel__', ignore_errors=True)

#------------------------------------------------------------------------------
def test_f1():
    @types('int')
    def f1(x = 1):
        y = x - 1
        return y

    f = epyccel(f1)

    # ...
    assert f(2) == f1(2)
    assert f() == f1()
    # ...
#------------------------------------------------------------------------------
def test_f2():
    @types('real [:]', 'int')
    def f5(x, m1 = 2):
        x[:] = 0.
        for i in range(0, m1):
            x[i] = i * 1.

    f = epyccel(f5)

    # ...
    m1 = 3

    x = np.zeros(m1)
    f(x)

    x_expected = np.zeros(m1)
    f5(x_expected)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

    f(x, m1 = m1)

    f5(x_expected, m1)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )


#------------------------------------------------------------------------------
def test_f3():
    @types('real','real')
    def f3(x = 1.5, y = 2.5):
        return x+y

    f = epyccel(f3)

    # ...
    assert f(19.2,6.7) == f3(19.2,6.7)
    assert f(4.5) == f3(4.5)
    assert f(y = 8.2) == f3(y = 8.2)
    assert f() == f3()
    # ...
##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================
#
#def teardown_module():
#    clean_test()
#
