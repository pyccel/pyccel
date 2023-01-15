# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

def functional_for_1d_range():
    a = [i for i in range(4)]
    return len(a), a[0], a[1], a[2], a[3]

def functional_for_overwrite_1d_range():
    a = [i for i in range(4)]
    a = [i for i in range(1,5)]
    return len(a), a[0], a[1], a[2], a[3]

@types('int[:]')
def functional_for_1d_var(y):
    a = [yi for yi in y]
    return len(a), a[0], a[1], a[2], a[3]

@types('int[:]', 'int')
def functional_for_1d_const(y,z):
    a = [z for _ in y]
    return len(a), a[0], a[1], a[2], a[3]

def functional_for_1d_const2():
    a = [5 for _ in range(0,4,2)]
    return len(a), a[0], a[1]

def functional_for_2d_range():
    a = [i*j for i in range(3) for j in range(2)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

@types('int[:]')
def functional_for_2d_var_range(y):
    a = [yi for yi in y for j in range(2)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

@types('int[:]','int[:]')
def functional_for_2d_var_var(y,z):
    a = [yi*zi for yi in y for zi in z]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_2d_dependant_range_1():
    a = [i*j for i in range(4) for j in range(i)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_2d_dependant_range_2():
    a = [i*j for i in range(3) for j in range(i,3)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_2d_dependant_range_3():
    a = [i*j for i in range(1,4) for j in range(0,4,i)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]

@types('int')
def functional_for_2d_array_range(idx):
    a = [(x1, y1, z1)  for x1 in range(3) for y1 in range(x1,5) for z1 in range(y1,10)]
    return len(a), a[idx][0], a[idx][1], a[idx][2]

def functional_for_2d_range_const():
    a = [20 for _ in range(3) for _ in range(2)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_3d_range():
    a = [i*j for i in range(1,3) for j in range(i,4) for k in range(i,j)]
    return len(a), a[0], a[1], a[2], a[3]
