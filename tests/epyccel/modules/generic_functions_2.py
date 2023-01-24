# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types
from pyccel.decorators import template


@template('Z', types=['int', 'real'])
@types('Z', 'Z')
def tmplt_1(x, y):
    return x + y

@template('Z', types=['int', 'real'])
@template('Y', types=['int', 'real'])
@types('Z', 'Z', 'Y')
def multi_tmplt_1(x, y, z):
    return x + y + z

@types('int', 'int')
@types('int', 'real')
def multi_heads_1(x, y):
    return x + y

@template(types=['int', 'real'], name = 'Z')
@types('Z', 'Z')
def tmplt_2(x, y):
    return x + y

@template('K', types=['int'])
@template('G', types=['int', 'real'])
@types('K', 'G')
def multi_tmplt_2(y, z):
    return y + z


#------------------------------------------------------

@template('K', types=['int'])
@template('G', types=['int', 'real'])
@types('G', 'K')
def default_var_1(x , y = 5):
    return x + y


@template('K', types=['complex'])
@template('G', types=['int', 'real'])
@types('G', 'K')
def default_var_2(x , y = 5j):
    return x + y

@template('K', types=['bool'])
@template('G', types=['int', 'real'])
@types('G', 'K')
def default_var_3(x , y = False):
    if y is True:
        return x
    return x - 1

@types('int', 'int')
@types('real', 'int')
def default_var_4(x, y = 5):
    return x + y


@template('K', types=['int'])
@template('G', types=['int', 'real'])
@types('G', 'K')
def optional_var_1(x , y = None):
    if y is None :
        return x
    return x + y


@template('K', types=['complex'])
@template('G', types=['int', 'real'])
@types('G', 'K')
def optional_var_2(x , y = None):
    if y is None :
        return x + 1j
    return x + y

@types('int', 'real')
@types('real', 'real')
def optional_var_3(x, y = None):
    if y is None:
        return x / 2.0
    return x / y


@types('complex', 'int')
@types('real', 'int')
def optional_var_4(x, y = None):
    if y is None:
        return x
    return x + y

#-------------------------------------------------

@template('T', types=['int', 'int32', 'int8', 'int16', 'int64'])
@types('T', 'T')
def int_types(x, y):
    return x + y


@template('T', types=['real', 'float32', 'float64'])
@types('T', 'T')
def float_types(x, y):
    return x + y

@template('T', types=['complex', 'complex64', 'complex128'])
@types('T', 'T')
def complex_types(x, y):
    return x + y


@template('G', types=['complex', 'int'])
@template('H', types=['int', 'float64'])
@template('K', types=['int16', 'bool'])
@types('G', 'H', 'K')
def mix_types_1(x, y, z):
    if z :
        return x + y
    return x - y


@template('T', types=['int64', 'int32', 'int16', 'real', 'complex', 'float32'])
@types('T', 'T')
def mix_types_2(x, y):
    if y != x:
        return y - x
    return -x


@template('T', types=['int', 'int32'])
@types('T', 'T')
def mix_types_3(x, y):
    if y != x:
        return y - x
    return -x


@template('T', types=['int64[:]', 'real[:]', 'complex[:]'])
@types('T', 'int')
def mix_array_1(x, a):
    x[:] += a


@types('complex[:]', 'complex[:]', 'int')
@types('real[:]', 'int64[:]', 'int')
def mix_array_2(x, y, a):
    x[:] += a
    y[:] -= a

@template('T', types=['int32[:]', 'int8[:]', 'int16[:]', 'int64[:]'])
@types('T', 'int')
def mix_int_array_1(x, a):
    x[:] += a

@template('T', types=['int[:]', 'int32[:]', 'int64[:]'])
@types('T', 'int')
def mix_int_array_2(x, a):
    x[:] += a

@template('T', types=['real[:]', 'float32[:]'])
@types('T', 'real')
def mix_float_array_1(x, a):
    x[:] *= a

@template('T', types=['complex[:]', 'complex64[:]'])
@types('T', 'real')
def mix_complex_array_1(x, a):
    x[:] *= a

#$ header function dup_header(real)
#$ header function dup_header(float64)
@types('float')
@types('float64')
def dup_header(a):
    return a

@template('T', types=[float,int])
def zeros_type(a : 'T'):
    from numpy import zeros
    x = zeros(10,dtype= type(a))
    return x[0]

@types('int')
@types('int[:]')
def scalar_or_array(a):
    return a+2
