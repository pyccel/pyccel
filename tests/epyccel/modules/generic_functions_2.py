# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

Z1 = TypeVar('Z1', int, float)
Z2 = TypeVar('Z2', int, float)
NIT_1D = TypeVar('NIT_1D', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]')
IT_1D = TypeVar('IT_1D', 'int[:]', 'int32[:]', 'int64[:]')
FT_1D = TypeVar('FT_1D', 'float[:]', 'float32[:]')
CT_1D = TypeVar('CT_1D', 'complex[:]', 'complex64[:]')
IT = TypeVar('IT', 'int', 'int32', 'int8', 'int16', 'int64')
FT = TypeVar('FT', 'float', 'float32', 'float64')
FT2 = TypeVar('FT2', 'float', 'float64')
CT = TypeVar('CT', 'complex', 'complex64', 'complex128')
B = bool
I = int
C = complex
IA = TypeVar('IA', 'int', 'int[:]')
T1 = TypeVar('T1', complex, int)
T2 = TypeVar('T2', int, 'float64')
T3 = TypeVar('T3', 'int16', bool)
T4 = TypeVar('T4', 'int64', 'int32', 'int16', 'float', 'complex', 'float32')
T5 = TypeVar('T5', 'int', 'int32')
T6 = TypeVar('T6', 'int64[:]', 'float[:]', 'complex[:]')
T7 = TypeVar('T7', 'int', 'float[:]', 'complex[:, :]')

def tmplt_1(x : Z1, y : Z1):
    return x + y

def multi_tmplt_1(x : Z1, y : Z1, z : Z2):
    return x + y + z

def multi_heads_1(x : int, y : 'int | float'):
    return x + y

def tmplt_2(x : Z1, y : Z1):
    return x + y

def multi_tmplt_2(y : I, z : Z1):
    return y + z


#------------------------------------------------------

def default_var_1(x : Z1, y  : I =  5):
    return x + y


def default_var_2(x : Z1, y  : C =  5j):
    return x + y

def default_var_3(x : Z1, y  : B =  False):
    if y is True:
        return x
    return x - 1

def default_var_4(x : 'int | float', y : int = 5):
    return x + y


def optional_var_1(x : Z1, y  : I =  None):
    if y is None :
        return x
    return x + y


def optional_var_2(x : Z1, y  : C =  None):
    if y is None :
        return x + 1j
    return x + y

def optional_var_3(x : 'int | float', y : float = None):
    if y is None:
        return x / 2.0
    return x / y


def optional_var_4(x : 'complex | float', y : int = None):
    if y is None:
        return x
    return x + y

#-------------------------------------------------

def int_types(x : IT, y : IT):
    return x + y


def float_types(x : FT, y : FT):
    return x + y

def complex_types(x : CT, y : CT):
    return x + y


def mix_types_1(x : T1, y : T2, z : T3):
    if z :
        return x + y
    return x - y


def mix_types_2(x : T4, y : T4):
    if y != x:
        return y - x
    return -x


def mix_types_3(x : T5, y : T5):
    if y != x:
        return y - x
    return -x


def mix_array_scalar(x : T6):
    x[:] += 1

def mix_array_1(x : T6, a : 'int'):
    x[:] += a


def mix_array_2(x : 'complex[:] | float[:]', y : 'complex[:] | int64[:]', a : int):
    x[:] += a
    y[:] -= a

def mix_int_array_1(x : NIT_1D, a : 'int'):
    x[:] += a

def mix_int_array_2(x : IT_1D, a : 'int'):
    x[:] += a

def mix_float_array_1(x : FT_1D, a : 'float'):
    x[:] *= a

def mix_complex_array_1(x : CT_1D, a : 'float'):
    x[:] *= a

def dup_header(a : FT2):
    return a

def zeros_type(a : Z1):
    from numpy import zeros
    x = zeros(10,dtype= type(a))
    return x[0]

def scalar_or_array(a : IA):
    return a+2

def add_scalars_or_arrays(a: T7, b: T7):
    return a + b + 1
