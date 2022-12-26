# pylint: disable=missing-function-docstring, missing-module-docstring/

import numpy as np
from numpy.random import randint, uniform

from modules import types
from pyccel.epyccel import epyccel

def test_int_default(language):
    f1 = types.test_int_default
    f2 = epyccel(f1, language = language)

    a = randint(low = -1e9, high = 0, dtype = int) # negative
    b = randint(low = 0, high = 1e9, dtype = int) # positive

    assert f1(a) == f2(a) #add type comparaison when https://github.com/pyccel/pyccel/issues/735 is solved
    assert f1(b) == f2(b)


def test_int64(language):
    f1 = types.test_int64
    f2 = epyccel(f1, language = language)

    a = randint(low = -1e9, high = 0, dtype = np.int64) # negative
    b = randint(low = 0, high = 1e9, dtype = np.int64) # positive

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_int32(language):
    f1 = types.test_int32
    f2 = epyccel(f1, language = language)

    a = randint(low = -1e9, high = 0, dtype = np.int32) # negative
    b = randint(low = 0, high = 1e9, dtype = np.int32) # positive

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_int16(language):
    f1 = types.test_int16
    f2 = epyccel(f1, language = language)

    a = randint(low = -32768, high = 0, dtype = np.int16) # negative
    b = randint(low = 0, high = 32767, dtype = np.int16) # positive

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_int8(language):
    f1 = types.test_int8
    f2 = epyccel(f1, language = language)

    a = randint(low = -128, high = 0, dtype = np.int8) # negative
    b = randint(low = 0, high = 127, dtype = np.int8) # positive

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_real_defaultl(language):
    f1 = types.test_real_default
    f2 = epyccel(f1, language = language)

    a = uniform() * 1e9 # negative
    b = uniform() * -1e9 # positive

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_float32(language):
    f1 = types.test_float32
    f2 = epyccel(f1, language = language)

    a = np.float32(uniform() * 1e9) # negative
    b = np.float32(uniform() * -1e9) # positive

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_float64(language):
    f1 = types.test_float64
    f2 = epyccel(f1, language = language)

    a = np.float64(uniform() * 1e9) # negative
    b = np.float64(uniform() * -1e9) # positive

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_complex_default(language):
    f1 = types.test_complex_default
    f2 = epyccel(f1, language = language)

    a = complex(uniform() * -1e9, uniform() * 1e9)
    b = complex(uniform() * 1e9, uniform() * -1e9)

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_complex64(language):
    f1 = types.test_complex64
    f2 = epyccel(f1, language = language)

    a = complex(uniform() * -1e9, uniform() * 1e9)
    b = complex(uniform() * 1e9, uniform() * -1e9)

    a = np.complex64(a)
    b = np.complex64(b)

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_complex128(language):
    f1 = types.test_complex128
    f2 = epyccel(f1, language = language)

    a = complex(uniform() * -1e9, uniform() * 1e9)
    b = complex(uniform() * 1e9, uniform() * -1e9)

    a = np.complex128(a)
    b = np.complex128(b)

    assert f1(a) == f2(a)
    assert f1(b) == f2(b)

def test_bool(language):
    f1 = types.test_bool
    f2 = epyccel(f1, language = language)

    assert f1(True) == f2(True)
    assert f1(False) == f2(False)
