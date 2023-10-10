# pylint: disable=missing-function-docstring, missing-module-docstring
import platform
from pyccel.epyccel import epyccel

def test_or_boolean(language):
    def or_bool(a : 'bool', b : 'bool'):
        c = False
        if (a):
            c = True
        if (b):
            c = True
        return c
    epyc_or_bool = epyccel(or_bool, language=language)

    assert(epyc_or_bool(True,True)==or_bool(True,True))
    assert(epyc_or_bool(True,False)==or_bool(True,False))
    assert(epyc_or_bool(False,False)==or_bool(False,False))

def test_real_greater_bool(language):
    def real_greater_bool(x0 : 'float', x1 : 'float'):
        greater = False
        if x0 > x1:
            greater = True
        return greater

    epyc_real_greater_bool = epyccel(real_greater_bool, language=language)

    assert(real_greater_bool(1.0,2.0)==epyc_real_greater_bool(1.0,2.0))
    assert(real_greater_bool(1.5,1.2)==epyc_real_greater_bool(1.5,1.2))

def test_input_output_matching_types(language):
    def add_real(a : 'float', b : 'float'):
        c = a+b
        return c

    fflags="-Werror -Wconversion"
    if language=="fortran":
        fflags=fflags+"-extra"
    if platform.system() == 'Darwin' and language=='c': # If macosx
        fflags=fflags+" -Wno-error=unused-command-line-argument"
    epyc_add_real = epyccel(add_real, fflags=fflags, language=language)

    assert(add_real(1.0,2.0)==epyc_add_real(1.0,2.0))

def test_output_types_1(language):
    def cast_to_int(a : 'float'):
        b = int(a)
        return b

    f = epyccel(cast_to_int, language = language)
    assert(type(cast_to_int(5.2)) == type(f(5.2))) # pylint: disable=unidiomatic-typecheck

def test_output_types_2(language):
    def cast_to_float(a : 'int'):
        b = float(a)
        return b

    f = epyccel(cast_to_float,language= language)
    assert(type(cast_to_float(5)) == type(f(5)))    # pylint: disable=unidiomatic-typecheck

def test_output_types_3(language):
    def cast_to_bool(a : 'int'):
        b = bool(a)
        return b

    f = epyccel(cast_to_bool, language=language)
    assert(cast_to_bool(1) == f(1))

