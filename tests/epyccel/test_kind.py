from pyccel.decorators import types
from pyccel import epyccel
import shutil


def clean_test():
    shutil.rmtree('__pycache__', ignore_errors=True)
    shutil.rmtree('__epyccel__', ignore_errors=True)

def test_or_boolean():
    @types('bool', 'bool')
    def or_bool(a, b):
        c = False
        if (a):
            c = True
        if (b):
            c = True
        return c
    epyc_or_bool = epyccel(or_bool)

    assert(epyc_or_bool(True,True)==or_bool(True,True))
    assert(epyc_or_bool(True,False)==or_bool(True,False))
    assert(epyc_or_bool(False,False)==or_bool(False,False))

def test_real_greater_bool():
    @types('real', 'real')
    def real_greater_bool(x0, x1):
        greater = False
        if x0 > x1:
            greater = True
        return greater

    epyc_real_greater_bool = epyccel(real_greater_bool)

    assert(real_greater_bool(1.0,2.0)==epyc_real_greater_bool(1.0,2.0))
    assert(real_greater_bool(1.5,1.2)==epyc_real_greater_bool(1.5,1.2))

def test_input_output_matching_types():
    @types('float32', 'float32')
    def add_real(a, b):
        c = a+b
        return c

    epyc_add_real = epyccel(add_real, fflags="-Werror -Wconversion-extra")

    assert(add_real(1.0,2.0)==epyc_add_real(1.0,2.0))

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module():
    clean_test()

