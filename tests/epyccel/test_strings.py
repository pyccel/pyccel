# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect
import pytest

from pyccel import epyccel
from modules import strings_module

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
    assert python_out == pyccel_out.strip()
