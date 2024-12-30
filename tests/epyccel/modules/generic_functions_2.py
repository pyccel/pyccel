# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import template, types


@template('Z', types=['int', 'float'])
def tmplt_1(x : 'Z', y : 'Z'):
    return x + y

@template('Z', types=['int', 'float'])
@template('Y', types=['int', 'float'])
def multi_tmplt_1(x : 'Z', y : 'Z', z : 'Y'):
    return x + y + z

def multi_heads_1(x : int, y : 'int | float'):
    return x + y

@template(types=['int', 'float'], name = 'Z')
def tmplt_2(x : 'Z', y : 'Z'):
    return x + y

@template('K', types=['int'])
@template('G', types=['int', 'float'])
def multi_tmplt_2(y : 'K', z : 'G'):
    return y + z


#------------------------------------------------------

@template('K', types=['int'])
@template('G', types=['int', 'float'])
def default_var_1(x : 'G', y  : 'K' =  5):
    return x + y


@template('K', types=['complex'])
@template('G', types=['int', 'float'])
def default_var_2(x : 'G', y  : 'K' =  5j):
    return x + y

@template('K', types=['bool'])
@template('G', types=['int', 'float'])
def default_var_3(x : 'G', y  : 'K' =  False):
    if y is True:
        return x
    return x - 1

def default_var_4(x : 'int | float', y : int = 5):
    return x + y


@template('K', types=['int'])
@template('G', types=['int', 'float'])
def optional_var_1(x : 'G', y  : 'K' =  None):
    if y is None :
        return x
    return x + y


@template('K', types=['complex'])
@template('G', types=['int', 'float'])
def optional_var_2(x : 'G', y  : 'K' =  None):
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

@template('T', types=['int', 'int32', 'int8', 'int16', 'int64'])
def int_types(x : 'T', y : 'T'):
    return x + y


@template('T', types=['float', 'float32', 'float64'])
def float_types(x : 'T', y : 'T'):
    return x + y

@template('T', types=['complex', 'complex64', 'complex128'])
def complex_types(x : 'T', y : 'T'):
    return x + y


@template('G', types=['complex', 'int'])
@template('H', types=['int', 'float64'])
@template('K', types=['int16', 'bool'])
def mix_types_1(x : 'G', y : 'H', z : 'K'):
    if z :
        return x + y
    return x - y


@template('T', types=['int64', 'int32', 'int16', 'float', 'complex', 'float32'])
def mix_types_2(x : 'T', y : 'T'):
    if y != x:
        return y - x
    return -x


@template('T', types=['int', 'int32'])
def mix_types_3(x : 'T', y : 'T'):
    if y != x:
        return y - x
    return -x


@template('T', types=['int64[:]', 'float[:]', 'complex[:]'])
def mix_array_scalar(x : 'T'):
    x[:] += 1

@template('T', types=['int64[:]', 'float[:]', 'complex[:]'])
def mix_array_1(x : 'T', a : 'int'):
    x[:] += a


def mix_array_2(x : 'complex[:] | float[:]', y : 'complex[:] | int64[:]', a : int):
    x[:] += a
    y[:] -= a

@template('T', types=['int32[:]', 'int8[:]', 'int16[:]', 'int64[:]'])
def mix_int_array_1(x : 'T', a : 'int'):
    x[:] += a

@template('T', types=['int[:]', 'int32[:]', 'int64[:]'])
def mix_int_array_2(x : 'T', a : 'int'):
    x[:] += a

@template('T', types=['float[:]', 'float32[:]'])
def mix_float_array_1(x : 'T', a : 'float'):
    x[:] *= a

@template('T', types=['complex[:]', 'complex64[:]'])
def mix_complex_array_1(x : 'T', a : 'float'):
    x[:] *= a

@template('T', ['float', 'float64'])
def dup_header(a : 'T'):
    return a

@template('T', types=[float,int])
def zeros_type(a : 'T'):
    from numpy import zeros
    x = zeros(10,dtype= type(a))
    return x[0]

@template('T', ['int', 'int[:]'])
def scalar_or_array(a : 'T'):
    return a+2

@template('T', types=['int', 'float[:]', 'complex[:, :]'])
def add_scalars_or_arrays(a: 'T', b: 'T'):
    return a + b + 1
