# pylint: disable=missing-function-docstring, missing-module-docstring
# TODO add AugAssign to these tests

def f_assign(m1 : int, x : 'double[:]'):
    x[:] = 0.
    for i in range(0, m1):
        x[i] = i * 1.

def f_for(m1 : int, x : 'double[:]'):
    for i in range(0, m1):
        x[i] = i * 1.

def f_if(m1 : int, x : 'double[:]'):
    if m1 == 1:
        x[1] = 1.

def f_while(m1 : int, x : 'double[:]'):
    i = 0
    while(i < 10):
        x[i] = i * 1.
        i = i + 1
