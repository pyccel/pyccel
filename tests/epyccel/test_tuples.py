# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect
import pytest
import numpy as np

from pyccel.epyccel import epyccel
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
        'homogeneous_tuple_string':"Can't save a list of strings (#459)",
        'tuple_homogeneous_return':"Can't return a tuple",
        'tuple_inhomogeneous_return':"Can't return a tuple",
        'tuple_visitation_inhomogeneous':"Can't iterate over an inhomogeneous tuple",
        'tuple_homogeneous_string':"Can't save a list of strings (#459)",
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

        if isinstance(pth, np.ndarray):
            assert np.allclose(pth,pycc)

        elif isinstance(pth, bool):
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

def test_homogeneous_tuples_of_bools_as_args(language):
    def my_tuple(a : 'tuple[bool,...]'):
        return len(a), a[0], a[1], a[2]

    epyc_func = epyccel(my_tuple, language=language)
    assert my_tuple((True, False, False)) == epyc_func((True, False, False))
    tuple_arg = (False, True, False, True, True, True)
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)

def test_homogeneous_tuples_of_ints_as_args(language):
    def my_tuple(a : 'tuple[int,...]'):
        return len(a), a[0], a[1], a[2]

    epyc_func = epyccel(my_tuple, language=language)
    assert my_tuple((1,2,3)) == epyc_func((1,2,3))
    tuple_arg = (-1, 9, 20, -55, 23)
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)

def test_homogeneous_tuples_of_floats_as_args(language):
    def my_tuple(a : 'tuple[float,...]'):
        return len(a), a[0], a[1], a[2]

    epyc_func = epyccel(my_tuple, language=language)
    assert my_tuple((1.0,2.0,3.0)) == epyc_func((1.0,2.0,3.0))
    tuple_arg = (-1.0, 9.0, 20.0, -55.3, 23.2)
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)

def test_homogeneous_tuples_of_complexes_as_args(language):
    def my_tuple(a : 'tuple[complex,...]'):
        return len(a), a[0], a[1], a[2]

    epyc_func = epyccel(my_tuple, language=language)
    assert my_tuple((1.0+4j, 2.0-2j, 3.0+0j)) == epyc_func((1.0+4j, 2.0-2j, 3.0+0j))
    tuple_arg = (1.0+4j, 2.0-2j, 3.0+0j, -23.12-4.4j)
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)

def test_homogeneous_tuples_of_numpy_ints_as_args(language):
    def my_tuple(a : 'tuple[int8,...]'):
        return len(a), a[0], a[1], a[2]

    epyc_func = epyccel(my_tuple, language=language)
    tuple_arg = (np.int8(1), np.int8(2), np.int8(3))
    assert my_tuple(tuple_arg) == epyc_func(tuple_arg)
