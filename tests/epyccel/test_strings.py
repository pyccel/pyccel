# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect
import pytest

from modules import strings_module
from pyccel import epyccel

string_funcs = [getattr(strings_module,f) for f in strings_module.__all__ if inspect.isfunction(getattr(strings_module,f))]

failing_tests = {
        'concatenate':'C does not support string concatenation',
        'concatenate_multiple':'C does not support string concatenation',
        'concatenate_expr':'C does not support string concatenation',
        }

@pytest.mark.parametrize('test_func', string_funcs)
def test_strings(test_func, language):
    if test_func.__name__ in failing_tests and language=='c':
        pytest.xfail(failing_tests[test_func.__name__])

    f1 = test_func
    f2 = epyccel( f1, language = language )

    python_out = f1()
    pyccel_out = f2()
    print(python_out)
    print(pyccel_out)
    assert python_out == pyccel_out

def test_string_compare(language):
    def str_comp():
        a = 'hello'
        if a == 'world':
            return 1
        elif a != 'boo':
            return 2
        elif a == 'hello':
            return 3
        else:
            return 4

    f = epyccel( str_comp, language=language )

    assert str_comp() == f()

def test_string_argument(language):
    def str_option_test(option : str):
        if option == 'do this':
            return 1.0
        else:
            return 2.0

    f = epyccel( str_option_test, language=language )

    assert str_option_test('do this') == f('do this')
    assert str_option_test('do that') == f('do that')

def test_string_argument_optional(language):
    def str_option_test(option : str = None):
        if option is not None and option == 'do this':
            return 1.0
        else:
            return 2.0

    f = epyccel( str_option_test, language=language )

    assert str_option_test('do this') == f('do this')
    assert str_option_test('do that') == f('do that')
    assert str_option_test() == f()

