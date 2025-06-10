# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

T : type = 'int | float'
R : type = 'int | float'
S : type = 'int | float | complex'

def gen_1(a : 'float'):
    return a / 10

def gen_2(y : 'float | int', x : 'int | float'):
    return y / x

def gen_3(x : T, y : T):
    return x / y

def gen_4(x : T, y : T):
    return x / y

def gen_5(x : T, y : R):
    return x / y

def gen_6(x : S, y : S):
    return x + y

def gen_7(x : T, y : T, z : R):
    return x + y + z

Z = TypeVar('Z', int, float)
O = TypeVar('O', int, float)

def tmplt_head_1(x : Z, y : Z):
    return x + y

def local_override_1(x : O, y : O):
    return x + y

def tmplt_tmplt_1(x : Z, y : Z, z : R):
    return x + y + z

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
    z = gen_6(complex(1, 2), complex(1, 2))
    a = gen_6(5.22 + 3.14j, 0.15 + 12j)
    return x, y, z, a

def tst_gen_7():
    x = gen_7(5, 5, 7)
    y = gen_7(5, 5, 7.3)
    z = gen_7(4.5, 4.5, 8)
    a = gen_7(7.5, 3.5, 7.7)
    return x, a, y, z


def tst_tmplt_head_1():
    x = tmplt_head_1(5, 5)
    y = tmplt_head_1(5.5, 7.3)
    return x, y

def tst_local_override_1():
    x = local_override_1(5, 4)
    y = local_override_1(6.56, 3.3)
    return x, y

def tst_tmplt_tmplt_1():
    x = tmplt_tmplt_1(5, 5, 5)
    y = tmplt_tmplt_1(5.5, 7.3, 7.7)
    z = tmplt_tmplt_1(5.5, 5.56, 7)
    a = tmplt_tmplt_1(5, 5, 7.7)
    return x, y, z, a
