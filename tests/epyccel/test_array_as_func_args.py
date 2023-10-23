"""
This test file is made for testing the functionality of passing numpy arrays
in the function arguments.
"""
# pylint: disable=missing-function-docstring
import numpy as np
from numpy.random import randint, uniform

from pyccel.epyccel import epyccel
from pyccel.decorators import template

int_types = ['int8', 'int16', 'int32', 'int64']
float_types = ['float32', 'float64']

def test_array_int_1d_scalar_add(language):
    @template('T', ['int8', 'int16', 'int32', 'int64'])
    def array_int_1d_scalar_add(x : 'T[:]', a : 'T', x_len : int):
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

def test_array_real_1d_scalar_add(language):
    @template('T', ['float32', 'double'])
    def array_real_1d_scalar_add(x : 'T[:]', a : 'T', x_len : int):
        for i in range(x_len):
            x[i] += a
    f1 = array_real_1d_scalar_add
    f2 = epyccel(f1, language=language)

    for t in float_types:
        size = randint(1, 30)
        x1 = uniform(np.finfo(t).max / 2, size=size)
        x2 = np.copy(x1)
        a = uniform(np.finfo(t).max / 2)

        f1(x1, a, size)
        f2(x2, a, size)

        assert np.array_equal( x1, x2 )

def test_array_complex_1d_scalar_add(language):
    @template('T', ['complex64', 'complex128'])
    def array_complex_1d_scalar_add(x : 'T[:]', a : 'T', x_len : int):
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
    @template('T', ['int8', 'int16', 'int32', 'int64'])
    def array_int_2d_scalar_add( x : 'T[:,:]', a : 'T', d1 : int, d2 : int):
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

def test_array_real_2d_scalar_add(language):
    @template('T', ['float32', 'double'])
    def array_real_2d_scalar_add(x : 'T[:,:]', a : 'T', d1 : int, d2 : int):
        for i in range(d1):
            for j in range(d2):
                x[i, j] += a
    f1 = array_real_2d_scalar_add
    f2 = epyccel(f1, language=language)

    for t in float_types:
        d1 = randint(1, 15)
        d2 = randint(1, 15)
        x1 = uniform(np.finfo(t).max / 2, size=(d1, d2))
        x2 = np.copy(x1)
        a = uniform(np.finfo(t).max / 2)

        f1(x1, a, d1, d2)
        f2(x2, a, d1, d2)

        assert np.array_equal( x1, x2 )

def test_array_complex_2d_scalar_add(language):
    @template('T', ['complex64', 'complex128'])
    def array_complex_2d_scalar_add(x : 'T[:,:]', a : 'T', d1 : int, d2 : int):
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
