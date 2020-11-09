# pylint: disable=missing-function-docstring, missing-module-docstring/
import inspect
import pytest

from pyccel.epyccel import epyccel
from modules import strings as strings_module

string_funcs = [(f, getattr(strings_module,f)) for f in strings_module.__all__ if inspect.isfunction(getattr(strings_module,f))]

#failing_tests = {
#        'homogenous_string_string':'String has no precision',
#        'string_multi_indexing_1':'Multi object part of numpy array stored in sympy Tuple',
#        'string_multi_indexing_2':'Multi object part of numpy array stored in sympy Tuple',
#        'string_homogeneous_return':"Can't return a string",
#        'string_inhomogeneous_return':"Can't return a string",
#        'string_visitation_inhomogeneous':"Can't iterate over an inhomogeneous string",
#        }
failing_tests = dict()

marks = [f[1] if f[0] not in failing_tests else
        pytest.param(f[1], marks = pytest.mark.xfail(reason=failing_tests[f[0]])) for f in string_funcs]

@pytest.mark.skip(reason="Strings are arrays of chars. We cannot return arrays")
@pytest.mark.parametrize('test_func',marks)
def test_strings(test_func):
    f1 = test_func
    f2 = epyccel( f1 )

    python_out = f1()
    pyccel_out = (f2()).decode("utf-8")
    print(python_out)
    print(pyccel_out)
    assert(python_out == pyccel_out.strip())
