# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar
import numpy as np

T : type = 'int | float'
R : type = 'int | float'
O : type = 'bool | complex'
S : type = 'int | float'

Y = TypeVar('Y', int, float)
Z = TypeVar('Z', int, float)
K : type = int
G = TypeVar('G', int, float)
J = TypeVar('J', int, int)

def gen_1(a : float) -> float:
    return a * 10

def gen_2(y : 'float | int', x : 'int | float'):
    return y * x

def gen_3(x : T, y : T):
    return x - y

def gen_4(x : T, y : T):
    return x + y

def gen_5(x : T, y : R):
    return x + y

def gen_6(x : S, y : S):
    return x + y

def gen_7(x : T, y : T, z : R):
    return x + y + z

def multi_heads_1(x : int, y : 'int | float'):
    return x + y

def tmplt_1(x : Z, y : Z):
    return x + y

def multi_tmplt_1(x : Z, y : Z, z : Y):
    return x + y + z

def tmplt_head_1(x : Z, y : Z):
    return x + y

def local_override_1(x : 'Z', y : 'Z'):
    return x + y

def tmplt_tmplt_1(x : Z, y : Z, z : R):
    return x + y + z

def array_elem1(x : 'int64 [:] | float64[:]'):
    return x[0]

def multi_tmplt_2(y : K, z : G):
    return y + z

def dup_types_1(a : J):
    return a

def dup_types_2(a : 'int | int'):
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
    return x * y

def tst_local_override_1():
    x = local_override_1(5, 4)
    y = local_override_1(6.56, 3.3)
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
