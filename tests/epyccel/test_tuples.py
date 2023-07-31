# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect
import pytest
import numpy as np

from pyccel.epyccel import epyccel
from pyccel.decorators import types
from modules import tuples as tuples_module

def is_func_with_0_args(f):
    """ Test if name 'f' corresponds to an argument in the
    tuples module with no arguments
    """
    func = getattr(tuples_module,f)
    return inspect.isfunction(func) and len(inspect.signature(func).parameters)==0

tuple_funcs = [(f, getattr(tuples_module,f)) for f in tuples_module.__all__
                                            if is_func_with_0_args(f)]

failing_tests = {
        'homogenous_tuple_string':'String has no precision',
        'tuple_homogeneous_return':"Can't return a tuple",
        'tuple_inhomogeneous_return':"Can't return a tuple",
        'tuple_visitation_inhomogeneous':"Can't iterate over an inhomogeneous tuple",
        }

failing_c_tests = {
    'tuple_arg_unpacking':'Functions in functions not implemented in c',
    'tuples_func':'Functions in functions not implemented in c',
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
        pytest.param(f[1], marks = pytest.mark.xfail(reason=failing_tests[f[0]])) \
                for f in tuple_funcs if f[0] not in failing_c_tests]
@pytest.mark.parametrize('test_func', marks)
def test_tuples(test_func, language):
    f1 = test_func
    f2 = epyccel( f1, language=language )

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)

c_marks = [f[1] for f in tuple_funcs if f[0] in failing_c_tests]
@pytest.mark.parametrize('test_func', c_marks)
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="function in function is not implemented yet\
                in C language"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_tuples_c_fail(test_func, language):
    f1 = test_func
    f2 = epyccel( f1, language=language )

    python_out = f1()
    pyccel_out = f2()
    compare_python_pyccel(python_out, pyccel_out)

@pytest.mark.parametrize('test_func',
        [tuples_module.tuple_unpacking_3,
         tuples_module.tuple_unpacking_4]
)
def test_tuples_with_2d_args(test_func, language):
    f1 = test_func
    f2 = epyccel( f1, language=language )

    python_x = np.random.randint(100, size=(3,4))
    pyccel_x = python_x.copy()

    f1(python_x)
    f2(pyccel_x)
    np.allclose(python_x, pyccel_x)

