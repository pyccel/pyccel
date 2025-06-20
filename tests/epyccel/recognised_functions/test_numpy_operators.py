# pylint: disable=missing-function-docstring, missing-module-docstring
import os
from typing import TypeVar
from numpy.random import randint
import numpy as np
import pytest

from pyccel import epyccel

int_types = (bool, np.int8, np.int64)
IT = TypeVar('IT', *int_types)
IT2 = TypeVar('IT2', *int_types)

def test_numpy_bit_and(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a & b
    def g(a : 'IT[:,:,:]', b : 'IT2'):
        return a & b
    def h(a : 'IT', b : 'IT2[:,:,:]'):
        return a & b

    epyc_f = epyccel(f, language=language)
    epyc_g = epyccel(g, language=language)
    epyc_h = epyccel(h, language=language)

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 127, size=(2,3,4), dtype = t_y)

            z = f(x,y)
            epyc_z = epyc_f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 127, dtype = t_y)

            z2 = g(x,y)
            epyc_z2 = epyc_g(x,y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            z3 = h(y,x)
            epyc_z3 = epyc_h(y,x)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype

def test_numpy_bit_or(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a | b
    def g(a : 'IT[:,:,:]', b : 'IT2'):
        return a | b
    def h(a : 'IT', b : 'IT2[:,:,:]'):
        return a | b

    epyc_f = epyccel(f, language=language)
    epyc_g = epyccel(g, language=language)
    epyc_h = epyccel(h, language=language)

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 127, size=(2,3,4), dtype = t_y)

            z = f(x,y)
            epyc_z = epyc_f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 127, dtype = t_y)

            z2 = g(x,y)
            epyc_z2 = epyc_g(x,y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            z3 = h(y,x)
            epyc_z3 = epyc_h(y,x)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype

def test_numpy_bit_xor(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a ^ b
    def g(a : 'IT[:,:,:]', b : 'IT2'):
        return a ^ b
    def h(a : 'IT', b : 'IT2[:,:,:]'):
        return a ^ b

    epyc_f = epyccel(f, language=language)
    epyc_g = epyccel(g, language=language)
    epyc_h = epyccel(h, language=language)

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 127, size=(2,3,4), dtype = t_y)

            z = f(x,y)
            epyc_z = epyc_f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 127, dtype = t_y)

            z2 = g(x,y)
            epyc_z2 = epyc_g(x,y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            z3 = h(y,x)
            epyc_z3 = epyc_h(y,x)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype

def test_numpy_bit_lshift(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a << b
    def g(a : 'IT[:,:,:]', b : 'IT2'):
        return a << b
    def h(a : 'IT', b : 'IT2[:,:,:]'):
        return a << b

    epyc_f = epyccel(f, language=language)
    epyc_g = epyccel(g, language=language)
    epyc_h = epyccel(h, language=language)

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 32, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 5, size=(2,3,4), dtype = t_y)

            z = f(x,y)
            epyc_z = epyc_f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 5, dtype = t_y)

            z2 = g(x,y)
            epyc_z2 = epyc_g(x,y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            x = randint(2 if t_x is bool else 32, dtype = t_x)
            y = randint(2 if t_y is bool else 5, size=(2,3,4), dtype = t_y)

            z3 = h(x,y)
            epyc_z3 = epyc_h(x,y)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype

def test_numpy_bit_rshift(language):
    def f(a : 'IT[:,:,:]', b : 'IT2[:,:,:]'):
        return a >> b
    def g(a : 'IT[:,:,:]', b : 'IT2'):
        return a >> b
    def h(a : 'IT', b : 'IT2[:,:,:]'):
        return a >> b

    epyc_f = epyccel(f, language=language)
    epyc_g = epyccel(g, language=language)
    epyc_h = epyccel(h, language=language)

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 32, size=(2,3,4), dtype = t_x)
            y = randint(2 if t_y is bool else 5, size=(2,3,4), dtype = t_y)

            z = f(x,y)
            epyc_z = epyc_f(x,y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 5, dtype = t_y)

            z2 = g(x,y)
            epyc_z2 = epyc_g(x,y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            x = randint(2 if t_x is bool else 32, dtype = t_x)
            y = randint(2 if t_y is bool else 5, size=(2,3,4), dtype = t_y)

            z3 = h(x,y)
            epyc_z3 = epyc_h(x,y)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype

@pytest.mark.skipif(os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU') == 'intel',
        reason="Intel's invert implementation does not seem to match.")
def test_numpy_bit_invert(language):
    def f(a : 'IT[:,:,:]'):
        return ~a

    epyc_f = epyccel(f, language=language)

    for t_x in int_types:
        x = randint(2 if t_x is bool else 127, size=(2,3,4), dtype = t_x)

        z = f(x)
        epyc_z = epyc_f(x)
        assert np.array_equal(epyc_z, z)
        assert z.dtype is epyc_z.dtype
