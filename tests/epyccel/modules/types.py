# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types

def test_int_default(x : 'int'):
    return x

def test_int64(x : 'int64'):
    return x

def test_int32(x : 'int32'):
    return x

def test_int16(x : 'int16'):
    return x

def test_int8(x : 'int8'):
    return x

def test_real_default(x : 'float'):
    return x

def test_float32(x : 'float32'):
    return x

def test_float64(x : 'float64'):
    return x

def test_complex_default(x : 'complex'):
    return x

def test_complex64(x : 'complex64'):
    return x

def test_complex128(x : 'complex128'):
    return x

def test_bool(x : 'bool'):
    return x
