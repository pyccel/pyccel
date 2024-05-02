# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel.decorators import types
from pyccel.decorators import template

#$ header function gen_2(real, int)
#$ header function gen_2(int, real)
#$ header function gen_4(T, T)
#$ header function tmplt_head_1(int, real)
#$ header template T(int|real)
#$ header template R(int|real)
#$ header template O(bool|complex)
#$ header template S(int|real)

@types('real')
def gen_1(a):
    return a * 10

def gen_2(y, x):
    return y * x

@types('T', 'T')
def gen_3(x, y):
    return x - y

def gen_4(x, y):
    return x + y

@types('T', 'R')
def gen_5(x, y):
    return x + y

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

#$ header function array_elem1(int64 [:]|float64[:])
def array_elem1(x):
    return x[0]

@template('k', types=['int'])
@template('g', types=['int', 'real'])
@types('k', 'g')
def multi_tmplt_2(y, z):
    return y + z

@template('g', types=['int', 'int'])
@types('g')
def dup_types_1(a):
    return a

@types('int|int')
def dup_types_2(a):
    return a

def tst_gen_1():
    x = gen_1(5.5)
    return x

def tst_gen_2():
    x = gen_2(5.5, 5)
    y = gen_2(5, 5.5)
    return x + y

def tst_gen_3():
    x = gen_3(5, 5)
    y = gen_3(5.5, 5.5)
    return x + y

def tst_gen_4():
    x = gen_4(5.5, 5.5)
    y = gen_4(5, 5)
    return x + y

def tst_gen_5():
    x = gen_5(5.5, 5.5)
    y = gen_5(5, 5)
    z = gen_5(5.5, 5)
    a = gen_5(5, 6.6)
    return x * a * y * z

def tst_gen_6():
    x = gen_6(5.5, 5.5)
    y = gen_6(5, 5)
    return x * y

def tst_gen_7():
    x = gen_7(5, 5, 7)
    y = gen_7(5, 5, 7.3)
    z = gen_7(4.5, 4.5, 8)
    a = gen_7(7.5, 3.5, 7.7)
    return x * a * y * z

def tst_multi_heads_1():
    x = multi_heads_1(5, 5)
    y = multi_heads_1(5, 7.3)
    return x * y

def tst_tmplt_1():
    x = tmplt_1(5, 5)
    y = tmplt_1(5.5, 7.3)
    return x * y

def tst_multi_tmplt_1():
    x = multi_tmplt_1(5, 5, 7)
    y = multi_tmplt_1(5, 5, 7.3)
    z = multi_tmplt_1(4.5, 4.5, 8)
    a = multi_tmplt_1(7.5, 3.5, 7.7)
    return x * a * y * z

def tst_tmplt_head_1():
    x = tmplt_head_1(5, 5)
    y = tmplt_head_1(5.5, 7.3)
    z = tmplt_head_1(5, 5.56)
    return x * y * z

def tst_local_overide_1():
    x = local_overide_1(5, 4)
    y = local_overide_1(6.56, 3.3)
    return x * y

def tst_tmplt_tmplt_1():
    x = tmplt_tmplt_1(5, 5, 5)
    y = tmplt_tmplt_1(5.5, 7.3, 7.7)
    z = tmplt_tmplt_1(5.5, 5.56, 7)
    a = tmplt_tmplt_1(5, 5, 7.7)
    return x * y * z * a

def tst_array_elem1():
    x1 = np.array([1,2,3], dtype=np.int64)
    y = array_elem1(x1)
    x2 = np.array([1.3,2.4,3.4], dtype=np.float64)
    x = array_elem1(x2)
    return x * y

def tst_multi_tmplt_2():
    x = multi_tmplt_2(5, 5)
    y = multi_tmplt_2(5, 7.3)
    return x * y

def tst_dup_types_1():
    x = dup_types_1(5)
    return x

def tst_dup_types_2():
    x = dup_types_2(5)
    return x
