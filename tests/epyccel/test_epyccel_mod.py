# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
from numpy.random import randint, uniform
from numpy import allclose

from pyccel.decorators import types
from pytest_teardown_tools import run_epyccel, clean_test

# Relative and absolute tolerances for array comparisons in the form
# numpy.isclose(a, b, rtol, atol). Windows has larger round-off errors.
if sys.platform == 'win32':
    RTOL = 1e-13
    ATOL = 1e-14
else:
    RTOL = 2e-14
    ATOL = 1e-15

def test_modulo_int_int(language):
    @types(int, int)
    def modulo_i_i(x, y):
        return x % y, x % -y, -x % y, -x % -y, y % -y, -y % y

    f = run_epyccel(modulo_i_i, language=language)
    x = randint(0, 1e6)
    y = randint(1, 1e6)


    f_output = f(x, y)
    modulo_i_i_output = modulo_i_i(x, y)
    assert modulo_i_i_output == f_output
    assert isinstance(f_output, type(modulo_i_i_output))

def test_modulo_real_real(language):
    @types('real', 'real')
    def modulo_r_r(x, y):
        return x % y, x % -y, -x % y, -x % -y, y % -y, -y % y

    f = run_epyccel(modulo_r_r, language=language)
    x = uniform(low=0, high=1e6)
    y = uniform(low=1, high=1e2)

    f_output = f(x, y)
    modulo_r_r_output = modulo_r_r(x, y)
    assert allclose(f_output, modulo_r_r_output, rtol=RTOL, atol=ATOL)
    assert isinstance(f_output, type(modulo_r_r_output))

def test_modulo_real_int(language):
    @types('real', 'int')
    def modulo_r_i(x, y):
        return x % y, x % -y, -x % y, -x % -y, y % -y, -y % y

    f = run_epyccel(modulo_r_i, language=language)
    x = uniform(low=0, high=1e6)
    y = randint(low=1, high=1e6)


    f_output = f(x, y)
    modulo_r_i_output = modulo_r_i(x, y)
    assert allclose(f_output, modulo_r_i_output, rtol=RTOL, atol=ATOL)
    assert isinstance(f_output, type(modulo_r_i_output))

def test_modulo_int_real(language):
    @types('int', 'real')
    def modulo_i_r(x, y):
        return x % y, x % -y, -x % y, -x % -y, y % -y, -y % y

    f = run_epyccel(modulo_i_r, language=language)
    x = randint(0, 1e6)
    y = uniform(low=1, high=1e2)

    f_output = f(x, y)
    modulo_i_r_output = modulo_i_r(x, y)
    assert allclose(f_output, modulo_i_r_output, rtol=RTOL, atol=ATOL)
    assert isinstance(f_output, type(modulo_i_r_output))

def test_modulo_multiple(language):
    @types('int', 'real', 'int')
    def modulo_multiple(x, y, z):
        return x % y % z, -x % y % z, -x % -y % z, -x % -y % -z, \
               x % -y % z, x % -y % -z, x % y % -z, -x % y % -z, \
                   -y % y % y, y % -y % y, y % y % -y

    f = run_epyccel(modulo_multiple, language=language)
    x = randint(0, 1e6)
    y = uniform(low=1, high=1e4)
    z = randint(low=1, high=1e2)

    assert allclose(f(x, y, z), modulo_multiple(x, y, z), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x, y, z), type(modulo_multiple(x, y, z)))

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module(module):
    clean_test()
