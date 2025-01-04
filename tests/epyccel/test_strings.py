# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect
import pytest

from pyccel import epyccel
from modules import strings_module

string_funcs = [(f, getattr(strings_module,f)) for f in strings_module.__all__ if inspect.isfunction(getattr(strings_module,f))]

failing_tests = {
        'concatenate':'C does not support string concatenation',
        'concatenate_multiple':'C does not support string concatenation',
        'concatenate_expr':'C does not support string concatenation',
        }

marks = [f[1] if f[0] not in failing_tests else
        pytest.param(f[1], marks = pytest.mark.skip(reason=failing_tests[f[0]])) for f in string_funcs]

@pytest.mark.parametrize('test_func',marks)
def test_strings(test_func, language):
    f1 = test_func
    f2 = epyccel( f1, language = language )

    python_out = f1()
    pyccel_out = f2()
    print(python_out)
    print(pyccel_out)
    assert python_out == pyccel_out.strip()
