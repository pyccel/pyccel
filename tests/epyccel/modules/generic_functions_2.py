# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types
from pyccel.decorators import template

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

@template('g', types=['int', 'int32', 'int64', 'int8', 'int16'])
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
