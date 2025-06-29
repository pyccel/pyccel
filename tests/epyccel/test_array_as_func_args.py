"""
This test file is made for testing the functionality of passing numpy arrays
in the function arguments.
"""
# pylint: disable=missing-function-docstring
from typing import TypeVar, Final
import numpy as np
import pytest
from numpy.random import randint, uniform

from pyccel import epyccel

int_types = ['int8', 'int16', 'int32', 'int64']
float_types = ['float32', 'float64']
complex_types = ['complex64', 'complex128']

def test_array_int_1d_scalar_add(language):
    T = TypeVar('T', 'int8', 'int16', 'int32', 'int64')
    def array_int_1d_scalar_add(x : 'T[:]', a : T, x_len : int):
        for i in range(x_len):
            x[i] += a
    f1 = array_int_1d_scalar_add
    f2 = epyccel(f1, language=language)

    for t in int_types:
        size = randint(1, 30)
        x1 = randint(np.iinfo(t).max / 2, size=size, dtype=t)
        x2 = np.copy(x1)
        a = randint(np.iinfo(t).max / 2, dtype=t)

        f1(x1, a, size)
        f2(x2, a, size)

        assert np.array_equal( x1, x2 )

def test_array_float_1d_scalar_add(language):
    T = TypeVar('T', 'float32', 'float')
    def array_float_1d_scalar_add(x : 'T[:]', a : T, x_len : int):
        for i in range(x_len):
            x[i] += a
    f1 = array_float_1d_scalar_add
    f2 = epyccel(f1, language=language)

    for t in float_types:
        size = randint(1, 30)
        x1 = uniform(np.finfo(t).max / 2, size=size).astype(t)
        x2 = np.copy(x1)
        a = uniform(np.finfo(t).max / 2, size=1).astype(t)[0]

        f1(x1, a, size)
        f2(x2, a, size)

        assert np.array_equal( x1, x2 )

def test_array_complex_1d_scalar_add(language):
    T = TypeVar('T', 'complex64', 'complex128')
    def array_complex_1d_scalar_add(x : 'T[:]', a : T, x_len : int):
        for i in range(x_len):
            x[i] += a
    f1 = array_complex_1d_scalar_add
    f2 = epyccel(f1, language=language)

    for t in float_types:
        size = randint(1, 30)
        x1 = uniform(np.finfo(t).max / 4, size=size).astype(t) + \
                uniform(np.finfo(t).max / 4, size=size).astype(t) * 1j
        x2 = np.copy(x1)
        a = (uniform(np.finfo(t).max / 4,size=1).astype(t) + \
                uniform(np.finfo(t).max / 4,size=1).astype(t) * 1j)[0]

        f1(x1, a, size)
        f2(x2, a, size)

        assert np.array_equal( x1, x2 )

def test_array_int_2d_scalar_add(language):
    T = TypeVar('T', 'int8', 'int16', 'int32', 'int64')
    def array_int_2d_scalar_add( x : 'T[:,:]', a : T, d1 : int, d2 : int):
        for i in range(d1):
            for j in range(d2):
                x[i, j] += a
    f1 = array_int_2d_scalar_add
    f2 = epyccel(f1, language=language)

    for t in int_types:
        d1 = randint(1, 15)
        d2 = randint(1, 15)
        x1 = randint(np.iinfo(t).max / 2, size=(d1, d2), dtype=t)
        x2 = np.copy(x1)
        a = randint(np.iinfo(t).max / 2, dtype=t)

        f1(x1, a, d1, d2)
        f2(x2, a, d1, d2)

        assert np.array_equal( x1, x2 )

def test_array_float_2d_scalar_add(language):
    T = TypeVar('T', 'float32', 'float')
    def array_float_2d_scalar_add(x : 'T[:,:]', a : T, d1 : int, d2 : int):
        for i in range(d1):
            for j in range(d2):
                x[i, j] += a
    f1 = array_float_2d_scalar_add
    f2 = epyccel(f1, language=language)

    for t in float_types:
        d1 = randint(1, 15)
        d2 = randint(1, 15)
        x1 = uniform(np.finfo(t).max / 2, size=(d1, d2)).astype(t)
        x2 = np.copy(x1)
        a = uniform(np.finfo(t).max / 2, size=1).astype(t)[0]

        f1(x1, a, d1, d2)
        f2(x2, a, d1, d2)

        assert np.array_equal( x1, x2 )

def test_array_complex_2d_scalar_add(language):
    T = TypeVar('T', 'complex64', 'complex128')
    def array_complex_2d_scalar_add(x : 'T[:,:]', a : T, d1 : int, d2 : int):
        for i in range(d1):
            for j in range(d2):
                x[i, j] += a
    f1 = array_complex_2d_scalar_add
    f2 = epyccel(f1, language=language)

    for t in float_types:
        d1 = randint(1, 15)
        d2 = randint(1, 15)
        x1 = uniform(np.finfo(t).max / 4, size=(d1, d2)).astype(t) + \
                uniform(np.finfo(t).max / 4, size=(d1, d2)).astype(t) * 1j
        x2 = np.copy(x1)
        a = (uniform(np.finfo(t).max / 4,size=1).astype(t) + \
                uniform(np.finfo(t).max / 4,size=1).astype(t) * 1j)[0]

        f1(x1, a, d1, d2)
        f2(x2, a, d1, d2)

        assert np.array_equal( x1, x2 )

def test_array_final(language):
    def array_final(x : 'Final[float[:]]'):
        return x[0]

    f1 = array_final
    f2 = epyccel(f1, language=language)

    d = randint(1, 15)
    x1 = uniform(np.finfo(float).max, size=(d))
    assert f1(x1) == f2(x1)

@pytest.mark.parametrize('language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c', marks = [
        pytest.mark.skip(reason="STC does not handle arrays of size 0"),
        pytest.mark.c]
    ),
    pytest.param('python', marks = pytest.mark.python))
)
def test_array_empty(language):
    def array_empty(x : 'float[:,:]'):
        return x.size

    f1 = array_empty
    f2 = epyccel(f1, language=language)

    x1 = uniform(np.finfo(float).max, size=(0,3))
    assert f1(x1) == f2(x1)
