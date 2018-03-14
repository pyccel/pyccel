# coding: utf-8

from pyccel.epyccel import epyccel
import numpy as np

def test_simple():
    header = '#$ header procedure static decr(int) results(int)'
    def decr(x):
        y = x - 1
        return y

    f = epyccel(decr, header)

    y = f(3)
    assert(y == 2)

def test_array_1():
    header = '#$ header procedure static f_static(int [:]) results(int)'
    def f_static(x):
        y = x[0] - 1
        return y

    f = epyccel(f_static, header)

    x = np.array([3, 4, 5, 6], dtype=int)
    y = f(x)
    assert(y == 2)

    y = f([3, 4, 5, 6])
    assert(y == 2)


if __name__ == '__main__':
    test_simple()
    test_array_1()
