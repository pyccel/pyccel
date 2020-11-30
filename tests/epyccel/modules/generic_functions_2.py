# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types
from pyccel.decorators import template


@template('z', types=['int', 'real'])
@types('z', 'z')
def tmplt_1(x, y):
    return x + y

@template('z', types=['int', 'real'])
@template('y', types=['int', 'real'])
@types('z', 'z', 'y')
def multi_tmplt_1(x, y, z):
    return x + y + z

@types('int', 'int')
@types('int', 'real')
def multi_heads_1(x, y):
    return x + y

@template(types=['int', 'real'], name = 'z')
@types('z', 'z')
def tmplt_2(x, y):
    return x + y

@template('k', types=['int'])
@template('g', types=['int', 'real'])
@types('k', 'g')
def multi_tmplt_2(y, z):
    return y + z


#------------------------------------------------------

@template('k', types=['int'])
@template('g', types=['int', 'real'])
@types('g', 'k')
def default_var_1(x , y = 5):
    return x + y


@template('k', types=['complex'])
@template('g', types=['int', 'real'])
@types('g', 'k')
def default_var_2(x , y = 5j):
    return x + y

@template('k', types=['bool'])
@template('g', types=['int', 'real'])
@types('g', 'k')
def default_var_3(x , y = False):
    if y is True:
        return x
    return x - 1

@types('int', 'int')
@types('real', 'int')
def default_var_4(x, y = 5):
    return x + y


@template('k', types=['int'])
@template('g', types=['int', 'real'])
@types('g', 'k')
def optional_var_1(x , y = None):
    if y is None :
        return x
    return x + y


@template('k', types=['complex'])
@template('g', types=['int', 'real'])
@types('g', 'k')
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

@template('g', types=['int', 'int32', 'int8', 'int16', 'int64'])
@types('g', 'g')
def int_types(x, y):
    return x + y


@template('g', types=['real', 'float32', 'float64'])
@types('g', 'g')
def float_types(x, y):
    return x + y

@template('g', types=['complex', 'complex64', 'complex128'])
@types('g', 'g')
def complex_types(x, y):
    return x + y


@template('g', types=['complex', 'int'])
@template('h', types=['int', 'float64'])
@template('k', types=['int16', 'bool'])
@types('g', 'h', 'k')
def mix_types_1(x, y, z):
    if z :
        return x + y
    return x - y


@template('g', types=['int', 'int32', 'int16', 'real', 'complex', 'float32'])
@types('g', 'g')
def mix_types_2(x, y):
    if y != x:
        return y - x
    return -x


@template('g', types=['int[:]', 'real[:]', 'complex[:]'])
@types('g', 'int')
def mix_array_1(x, a):
    x[:] += a


@types('complex[:]', 'complex[:]', 'int')
@types('real[:]', 'int[:]', 'int')
def mix_array_2(x, y, a):
    x[:] += a
    y[:] -= a

@template('g', types=['int[:]', 'int8[:]', 'int16[:]', 'int32[:]'])
@types('g', 'int')
def mix_int_array_1(x, a):
    x[:] += a

@template('g', types=['real[:]', 'float32[:]'])
@types('g', 'real')
def mix_float_array_1(x, a):
    x[:] *= a

@template('g', types=['complex[:]', 'complex64[:]'])
@types('g', 'real')
def mix_complex_array_1(x, a):
    x[:] *= a

