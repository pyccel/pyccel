# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect
import pytest
import numpy as np

from modules import pointers as pointers_module
from modules import return_pointers
from pyccel import epyccel

pointers_funcs = [(f, getattr(pointers_module,f)) for f in pointers_module.__all__ if inspect.isfunction(getattr(pointers_module,f))]

def compare_python_pyccel( p_output, f_output ):
    if p_output is None:
        assert f_output is None
        return
    if not hasattr(p_output, '__len__'):
        p_output = [p_output]
    if not hasattr(f_output, '__len__'):
        f_output = [f_output]
    assert len(p_output) == len(f_output)

    for pth, pycc in zip(p_output, f_output):

        if isinstance(pth, bool):
            pycc_bool = (pycc == 1)
            assert pth == pycc_bool

        elif isinstance(pth, (int, str)):
            assert isinstance(pycc,type(pth))
            assert pth==pycc

        else:
            assert np.isclose(pth,pycc)

marks = [f[1] for f in pointers_funcs]

@pytest.mark.parametrize('test_func',marks)
def test_pointers(test_func, language):
    f1 = test_func
    f2 = epyccel( f1 , language=language)

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)

def test_return_pointers(language):
    f1 = return_pointers.return_ambiguous_pointer_to_argument
    f2 = epyccel( f1 , language=language)

    x = np.array([1,2,3,4])
    y = x.copy()

    python_out = f1(x)
    pyccel_out = f2(y)

    compare_python_pyccel(python_out, pyccel_out)

    assert python_out is x
    if language == 'python':
        assert pyccel_out is y
    else:
        assert pyccel_out.base is y
