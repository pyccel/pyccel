# pylint: disable=missing-function-docstring, missing-module-docstring
# TODO add AugAssign to these tests

#$ header function f_assign(int, double [:])
def f_assign(m1, x):
    x[:] = 0.
    for i in range(0, m1):
        x[i] = i * 1.

#$ header function f_for(int, double [:])
def f_for(m1, x):
    for i in range(0, m1):
        x[i] = i * 1.

#$ header function f_if(int, double [:])
def f_if(m1, x):
    if m1 == 1:
        x[1] = 1.

#$ header function f_while(int, double [:])
def f_while(m1, x):
    i = 0
    while(i < 10):
        x[i] = i * 1.
        i = i + 1
