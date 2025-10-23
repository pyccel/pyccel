# pylint: disable=missing-function-docstring, missing-module-docstring
import inspect
from typing import TypeVar
import pytest
import numpy as np

from modules import tuples as tuples_module

from pyccel import epyccel
from pyccel.errors.errors import PyccelError


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
        'tuple_visitation_inhomogeneous':"Can't iterate over an inhomogeneous tuple",
        'tuple_homogeneous_string':"Can't save a list of strings (#459)",
        }


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

        if isinstance(pth, np.ndarray):
            assert np.allclose(pth,pycc)

        elif isinstance(pth, bool):
            pycc_bool = (pycc == 1)
            assert pth == pycc_bool

        elif isinstance(pth, (int, str)):
            assert isinstance(pycc,type(pth))
            assert pth==pycc

        else:
            assert np.isclose(pth,pycc)

marks = [f[1] if f[0] not in failing_tests else
        pytest.param(f[1], marks = pytest.mark.xfail(reason=failing_tests[f[0]])) \
                for f in tuple_funcs]
@pytest.mark.parametrize('test_func', marks)
def test_tuples(test_func, language):
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

    python_x = np.array(np.random.randint(100, size=(3,4)), dtype=int)
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

def test_homogeneous_tuples_template_args(language):
    T = TypeVar('T', int, float)

    def my_tuple(a : tuple[T,...]):
        return len(a), a[0], a[1], a[2]

    epyc_func = epyccel(my_tuple, language=language)
    tuple_int_arg = (1, 2, 3)
    tuple_float_arg = (4., 5., 6.)

    int_pyth = my_tuple(tuple_int_arg)
    int_epyc = epyc_func(tuple_int_arg)
    assert int_pyth == int_epyc
    assert isinstance(int_epyc[1], int)

    float_pyth = my_tuple(tuple_float_arg)
    float_epyc = epyc_func(tuple_float_arg)
    assert float_pyth == float_epyc
    assert isinstance(float_epyc[1], float)

def test_multi_level_tuple_arg(language):
    def my_tuple(a : 'tuple[tuple[int,...],...]'):
        return len(a), len(a[0]), a[0][0], a[1][0], a[0][1], a[1][1]

    tuple_arg = ((1,2), (3,4))

    if language != 'python':
        # Raises an error because tuples inside tuples may have different lengths
        # This could be removed once lists are supported as the tuples could then
        # be stored in lists instead of arrays.
        with pytest.raises(PyccelError):
            _ = epyccel(my_tuple, language=language)
    else:
        epyc_func = epyccel(my_tuple, language=language)

        assert my_tuple(tuple_arg) == epyc_func(tuple_arg)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Returning tuples from functions requires a reorganisation of the return system. See #337"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_homogeneous_tuples_result(language):
    def my_tuple() -> 'tuple[int, ...]':
        a = (1,2,3,4,5)
        return a

    epyc_func = epyccel(my_tuple, language=language)

    assert my_tuple() == epyc_func()
