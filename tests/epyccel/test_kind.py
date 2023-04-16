# pylint: disable=missing-function-docstring, missing-module-docstring
import platform
from pyccel.decorators import types
from pytest_teardown_tools import run_epyccel, clean_test

def test_or_boolean(language):
    @types('bool', 'bool')
    def or_bool(a, b):
        c = False
        if (a):
            c = True
        if (b):
            c = True
        return c
    epyc_or_bool = run_epyccel(or_bool, language=language)

    assert(epyc_or_bool(True,True)==or_bool(True,True))
    assert(epyc_or_bool(True,False)==or_bool(True,False))
    assert(epyc_or_bool(False,False)==or_bool(False,False))

def test_real_greater_bool(language):
    @types('float', 'float')
    def real_greater_bool(x0, x1):
        greater = False
        if x0 > x1:
            greater = True
        return greater

    epyc_real_greater_bool = run_epyccel(real_greater_bool, language=language)

    assert(real_greater_bool(1.0,2.0)==epyc_real_greater_bool(1.0,2.0))
    assert(real_greater_bool(1.5,1.2)==epyc_real_greater_bool(1.5,1.2))

def test_input_output_matching_types(language):
    @types('float', 'float')
    def add_real(a, b):
        c = a+b
        return c

    fflags="-Werror -Wconversion"
    if language=="fortran":
        fflags=fflags+"-extra"
    if platform.system() == 'Darwin' and language=='c': # If macosx
        fflags=fflags+" -Wno-error=unused-command-line-argument"
    epyc_add_real = run_epyccel(add_real, fflags=fflags, language=language)

    assert(add_real(1.0,2.0)==epyc_add_real(1.0,2.0))

def test_output_types_1(language):
    @types('float')
    def cast_to_int(a):
        b = int(a)
        return b

    f = run_epyccel(cast_to_int, language = language)
    assert(type(cast_to_int(5.2)) == type(f(5.2))) # pylint: disable=unidiomatic-typecheck

def test_output_types_2(language):
    @types('int')
    def cast_to_float(a):
        b = float(a)
        return b

    f = run_epyccel(cast_to_float,language= language)
    assert(type(cast_to_float(5)) == type(f(5)))    # pylint: disable=unidiomatic-typecheck

def test_output_types_3(language):
    @types('int')
    def cast_to_bool(a):
        b = bool(a)
        return b

    f = run_epyccel(cast_to_bool, language=language)
    assert(cast_to_bool(1) == f(1))


##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module(module):
    clean_test()
