"""
This test file is made for testing the functionality of passing numpy arrays
in the function arguments.
"""
# pylint: disable=missing-function-docstring
import numpy as np
from numpy.random import randint, uniform

from pyccel.epyccel import epyccel
from pyccel.decorators import types

int_types = ['int8', 'int16', 'int32', 'int64']
float_types = ['float32', 'float64']

def test_array_int_1d_scalar_add(language):
    @types( 'int8[:]' , 'int8' , 'int')
    @types( 'int16[:]', 'int16', 'int')
    @types( 'int32[:]', 'int32', 'int')
    @types( 'int64[:]', 'int64', 'int')
    def array_int_1d_scalar_add( x, a, x_len ):
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
    @types( 'float32[:]', 'float32', 'int')
    @types( 'double[:]' , 'double' , 'int')
    def array_real_1d_scalar_add( x, a, x_len ):
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
    @types( 'complex64[:]' , 'complex64' , 'int')
    @types( 'complex128[:]' , 'complex128' , 'int')
    def array_complex_1d_scalar_add( x, a, x_len ):
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
    @types( 'int8[:,:]' , 'int8' , 'int', 'int')
    @types( 'int16[:,:]', 'int16', 'int', 'int')
    @types( 'int32[:,:]', 'int32', 'int', 'int')
    @types( 'int64[:,:]', 'int64', 'int', 'int')
    def array_int_2d_scalar_add( x, a, d1, d2 ):
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
    @types( 'float32[:,:]', 'float32', 'int', 'int')
    @types( 'double[:,:]' , 'double' , 'int', 'int')
    def array_real_2d_scalar_add( x, a, d1, d2 ):
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
    @types( 'complex64[:,:]' , 'complex64' , 'int', 'int')
    @types( 'complex128[:,:]' , 'complex128' , 'int', 'int')
    def array_complex_2d_scalar_add( x, a, d1, d2 ):
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
