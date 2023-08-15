# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import template

#$ header function gen_2(real, int)
#$ header function gen_2(int, real)
#$ header function gen_4(T, T)
#$ header function tmplt_head_1(int, real)
#$ header template T(int|real)
#$ header template R(int|real)
#$ header template O(real|complex)
#$ header template S(int|real|complex)

def gen_1(a : 'float'):
    return a / 10

def gen_2(y, x):
    return y / x

def gen_3(x : 'T', y : 'T'):
    return x / y

def gen_4(x, y):
    return x / y

def gen_5(x : 'T', y : 'R'):
    return x / y

def gen_6(x : 'S', y : 'S'):
    return x + y

def gen_7(x : 'T', y : 'T', z : 'R'):
    return x + y + z



@template('Z', types=['int', 'real'])
def tmplt_head_1(x : 'Z', y : 'Z'):
    return x + y

@template('O', types=['int', 'real'])
def local_overide_1(x : 'O', y : 'O'):
    return x + y

@template('Z', types=['int', 'real'])
def tmplt_tmplt_1(x : 'Z', y : 'Z', z : 'R'):
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
