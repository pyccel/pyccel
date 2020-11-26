# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types
from pyccel.decorators import template

#$ header function gen_2(real, int)
#$ header function gen_2(int, real)
#$ header function gen_4(T, T)
#$ header function tmplt_head_1(int, real)
#$ header template T(int|real)
#$ header template R(int|real)
#$ header template O(bool|complex)
#$ header template S(int|real|complex|bool)

@types('real')
def gen_1(a):
    return a / 10

def gen_2(y, x):
    return y / x

@types('T', 'T')
def gen_3(x, y):
    return x / y

def gen_4(x, y):
    return x / y

@types('T', 'R')
def gen_5(x, y):
    return x / y

@types('S', 'S')
def gen_6(x, y):
    return x + y

@types('T', 'T', 'R')
def gen_7(x, y, z):
    return x + y + z

@types('int', 'int')
@types('int', 'real')
def multi_heads_1(x, y):
    return x + y

@template('z', types=['int', 'real'])
@types('z', 'z')
def tmplt_1(x, y):
    return x + y

@template('z', types=['int', 'real'])
@template('y', types=['int', 'real'])
@types('z', 'z', 'y')
def multi_tmplt_1(x, y, z):
    return x + y + z

@template('z', types=['int', 'real'])
@types('z', 'z')
def tmplt_head_1(x, y):
    return x + y

@template('O', types=['int', 'real'])
@types('O', 'O')
def local_overide_1(x, y):
    return x + y

@template('z', types=['int', 'real'])
@types('z', 'z', 'R')
def tmplt_tmplt_1(x, y, z):
    return x + y + z

@template(types=['int', 'real'], name = 'z')
@types('z', 'z')
def tmplt_2(x, y):
    return x + y

@template('k', types=['int'])
@template('g', types=['int', 'real'])
@types('k', 'g')
def multi_tmplt_2(y, z):
    return y + z

def tst_gen_1():
    x = gen_1(5.5)
    return x

def tst_gen_2():
    x = gen_2(5.5, 5)
    y = gen_2(5, 5.5)
    return x, y

def tst_gen_3():
    x = gen_3(5, 5)
    y = gen_3(5.5, 5.5)
    return x, y

def tst_gen_4():
    x = gen_4(5.5, 5.5)
    y = gen_4(5, 5)
    return x, y

def tst_gen_5():
    x = gen_5(5.5, 5.5)
    y = gen_5(5, 5)
    z = gen_5(5.5, 5)
    a = gen_5(5, 6.6)
    return x, a, y, z

def tst_gen_6():
    x = gen_6(5.5, 5.5)
    y = gen_6(5, 5)
    z = gen_6(1j, 1j)
    return x, y, z

def tst_gen_7():
    x = gen_7(5, 5, 7)
    y = gen_7(5, 5, 7.3)
    z = gen_7(4.5, 4.5, 8)
    a = gen_7(7.5, 3.5, 7.7)
    return x, a, y, z

def tst_multi_heads_1():
    x = multi_heads_1(5, 5)
    y = multi_heads_1(5, 7.3)
    return x, y

def tst_tmplt_1():
    x = tmplt_1(5, 5)
    y = tmplt_1(5.5, 7.3)
    return x, y

def tst_multi_tmplt_1():
    x = multi_tmplt_1(5, 5, 7)
    y = multi_tmplt_1(5, 5, 7.3)
    z = multi_tmplt_1(4.5, 4.5, 8)
    a = multi_tmplt_1(7.5, 3.5, 7.7)
    return x, a, y, z

def tst_tmplt_head_1():
    x = tmplt_head_1(5, 5)
    y = tmplt_head_1(5.5, 7.3)
    z = tmplt_head_1(5, 5.56)
    return x, y, z

def tst_local_overide_1():
    x = local_overide_1(5, 4)
    y = local_overide_1(6.56, 3.3)
    return x, y

def tst_tmplt_tmplt_1():
    x = tmplt_tmplt_1(5, 5, 5)
    y = tmplt_tmplt_1(5.5, 7.3, 7.7)
    z = tmplt_tmplt_1(5.5, 5.56, 7)
    a = tmplt_tmplt_1(5, 5, 7.7)
    return x, y, z, a

def tst_tmplt_2():
    x = tmplt_2(5, 5)
    y = tmplt_2(5.5, 7.3)
    return x, y

def tst_multi_tmplt_2():
    x = multi_tmplt_2(5, 5)
    y = multi_tmplt_2(5, 7.3)
    return x, y

#--------------------------------------------------
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
