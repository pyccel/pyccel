# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect
import pytest
import numpy as np

from pyccel.epyccel import epyccel
from modules import pointers as pointers_module
from pyccel.decorators import types

pointers_funcs = [(f, getattr(pointers_module,f)) for f in pointers_module.__all__ if inspect.isfunction(getattr(pointers_module,f))]

def compare_python_pyccel( p_output, f_output ):
    if p_output is None:
        assert(f_output is None)
        return
    if not hasattr(p_output, '__len__'):
        p_output = [p_output]
    if not hasattr(f_output, '__len__'):
        f_output = [f_output]
    assert(len(p_output) == len(f_output))

    for pth, pycc in zip(p_output, f_output):

        if isinstance(pth, bool):
            pycc_bool = (pycc == 1)
            assert(pth == pycc_bool)

        elif isinstance(pth, (int, str)):
            assert(isinstance(pycc,type(pth)))
            assert(pth==pycc)

        else:
            assert(np.isclose(pth,pycc))

marks = [f[1] for f in pointers_funcs]

@pytest.mark.parametrize('test_func',marks)
def test_pointers(test_func, language):
    f1 = test_func
    f2 = epyccel( f1 , language=language)

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)
