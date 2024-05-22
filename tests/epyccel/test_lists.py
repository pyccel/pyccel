# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect
import pytest

from pyccel import epyccel
from modules import lists as lists_module

list_funcs = [(f, getattr(lists_module,f)) for f in lists_module.__all__ if inspect.isfunction(getattr(lists_module,f))]

marks = [f[1] for f in list_funcs]

@pytest.mark.parametrize('test_func',marks)
@pytest.mark.parametrize( 'language', (pytest.param("python", marks = pytest.mark.python),))
def test_lists(test_func, language):
    f1 = test_func
    f2 = epyccel( f1 , language=language)

    python_out = f1()
    pyccel_out = f2()
    print(pyccel_out)
    print(python_out)

    assert python_out == pyccel_out
