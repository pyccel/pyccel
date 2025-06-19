# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar
from numpy.random import randint
import numpy as np

from pyccel import epyccel

int_types = (bool, np.int8, np.int32, np.int64)
IT = TypeVar('IT', *int_types)
IT2 = TypeVar('IT2', *int_types)

def test_numpy_bit_and(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a & b

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 127, size=(2,3,4), dtype = t_y)

            epyc_f = epyccel(f, language=language)

            z = f(x,y)
            epyc_z = f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

def test_numpy_bit_or(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a | b

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 127, size=(2,3,4), dtype = t_y)

            epyc_f = epyccel(f, language=language)

            z = f(x,y)
            epyc_z = f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

def test_numpy_bit_xor(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a ^ b

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 127, size=(2,3,4), dtype = t_y)

            epyc_f = epyccel(f, language=language)

            z = f(x,y)
            epyc_z = f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

def test_numpy_bit_lshift(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a << b

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 127, size=(2,3,4), dtype = t_y)

            epyc_f = epyccel(f, language=language)

            z = f(x,y)
            epyc_z = f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

def test_numpy_bit_rshift(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a >> b

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 127, size=(2,3,4), dtype = t_y)

            epyc_f = epyccel(f, language=language)

            z = f(x,y)
            epyc_z = f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

def test_numpy_bit_invert(language):
    def f(a : 'IT[:,:,:]'):
        return ~a

    for t_x in int_types:
        x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)

        epyc_f = epyccel(f, language=language)

        z = f(x)
        epyc_z = f(x)
        assert np.array_equal(epyc_z, z)
        assert z.dtype is epyc_z.dtype
