# pylint: disable=missing-function-docstring, missing-module-docstring/
import inspect
import pytest
import numpy as np

from pyccel.epyccel import epyccel
from modules import tuples as tuples_module

tuple_funcs = [(f, getattr(tuples_module,f)) for f in tuples_module.__all__ if inspect.isfunction(getattr(tuples_module,f))]

failing_tests = {
        'homogenous_tuple_string':'String has no precision',
        'tuple_multi_indexing_1':'Multi object part of numpy array stored in sympy Tuple',
        'tuple_multi_indexing_2':'Multi object part of numpy array stored in sympy Tuple',
        'tuple_homogeneous_return':"Can't return a tuple",
        'tuple_inhomogeneous_return':"Can't return a tuple",
        'tuple_visitation_inhomogeneous':"Can't iterate over an inhomogeneous tuple",
        }

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

marks = [f[1] if f[0] not in failing_tests else
        pytest.param(f[1], marks = pytest.mark.xfail(reason=failing_tests[f[0]])) for f in tuple_funcs]

@pytest.mark.parametrize('test_func',marks)
def test_tuples(test_func):
    f1 = test_func
    f2 = epyccel( f1 )

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)
