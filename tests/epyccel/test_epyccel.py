# coding: utf-8

from pyccel.epyccel import epyccel
import numpy as np

def test_simple():
    header = '#$ header procedure decr(int) results(int)'
    def decr(x):
        y = x - 1
        return y

    f = epyccel(decr, header)

    y = f(3)
    assert(y == 2)

def test_array_1():
    header = '#$ header procedure f1(int [:]) results(int)'
    def f1(x):
        y = x[0] - 1
        return y

    f = epyccel(f1, header)

    x = np.array([3, 4, 5, 6], dtype=int)
    y = f(x)
    assert(y == 2)

    y = f([3, 4, 5, 6])
    assert(y == 2)

def test_array_2():
    header = '#$ header procedure f2(int [:]) results(int [:])'
    def f2(x):
        y = x - 1
        return y

    f = epyccel(f2, header)

    x = np.array([3, 4, 5, 6], dtype=int)
    y = f(x)
    assert(np.allclose(y, np.array([2, 3, 4, 5])))

def test_array_3():
    header = '#$ header procedure g(double [:,:]) results(double [:,:])'
    def g(x):
        y = x - 1.0
        return y

    f = epyccel(g, header)

    x = np.random.random((2, 3))
    y = f(x)

def test_array_4():
    header = '#$ header procedure f1_py(int, double [:])'
    def f1_py(m1, x):
        x[:] = 0.
        for i in range(0, m1):
            x[i] = i * 1.

    f = epyccel(f1_py, header)

    m1 = 3
    x = np.zeros(m1)
    f(m1, x)

    x_expected = np.array([0., 1., 2.])
    assert(np.allclose(x, x_expected))

def test_array_5():
    header = '#$ header procedure f2_py(int, int, float [:,:])'
    def f2_py(m1, m2, x):
        x[:,:] = 0.
        for i in range(0, m1):
            for j in range(0, m2):
                x[i,j] = (i+j) * 1.

    f = epyccel(f2_py, header)

    m1 = 2
    m2 = 3
    x = np.zeros((m1,m2), order='F')
    f(m1, m2, x)

    # ... expected
    x_expected = np.zeros((m1,m2), order='F')
    for i in range(0, m1):
        for j in range(0, m2):
            x_expected[i,j] = (i+j) * 1.
    # ...
    assert(np.allclose(x, x_expected))


if __name__ == '__main__':
    test_simple()
    test_array_1()
    test_array_2()
    test_array_3()
    test_array_4()
    test_array_5()
