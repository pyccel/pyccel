# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np
from pyccel.decorators import allow_negative_index, types

def array_view():
    a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a[:, :]
    for i in range(2):
        for j in range(5):
            print(b[i][j])

def array_view_negative_literal_step():
    a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a[:, ::-1]
    for i in range(2):
        for j in range(5):
            print(b[i][j])

def array_view_negative_literal_step__2():
    a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a[:, 4:2:-1]
    for i in range(2):
        for j in range(2):
            print(b[i][j])

def array_view_negative_literal_step__3():
    a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    b = a[:, 3::-2]
    for i in range(2):
        for j in range(2):
            print(b[i][j])

@allow_negative_index('a')
def array_view_negative_variable_step():
    a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    i = -1
    b = a[::i, :]
    for i in range(2):
        for j in range(5):
            print(b[i][j])

@allow_negative_index('a')
def array_view_negative_var():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 9, 5]])
    v = -1
    x = a[v,1:]
    for i in range(2):
        print(x[i])

@allow_negative_index('a')
def array_view_negative_literal():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 9, 5]])
    x = a[-1,1:]
    for i in range(2):
        print(x[i])

def array_view_positive_literal():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 9, 5]])
    y = a[1:, 1:3]
    for i in range(2):
        for j in range(2):
            print(y[i][j])

def array_view_2():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[1:-1]
    c = b[1:-1]
    for i in range(np.shape(c)[0]):
        print(c[i])

def array_view_3():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = a[1:-1]
    c = b[1:-1]
    c = a
    for i in range(np.shape(c)[0]):
        print(c[i])

@types('int[:]', 'int')
def array_1d_view_assign(x, a):
    x[:] = a

@types('int[:, :]', 'int')
def array_2d_view_assign(x, a):
    x[:, :] = a

@types('int[:, :, :]', 'int')
def array_3d_view_assign(x, a):
    x[:, :, :] = a

def array_1d_view():
    x = np.zeros((10,), dtype=int)
    array_1d_view_assign(x[:], 1)
    array_1d_view_assign(x[:6:], 2)
    array_1d_view_assign(x[6::], 3)
    array_1d_view_assign(x[2:3], 4)
    for i in range(np.shape(x)[0]):
        print(x[i])

def array_2d_view():
    x = np.zeros((5, 5), dtype=int)
    array_2d_view_assign(x[::, ::], 9)
    array_2d_view_assign(x[:2:2, :2:3], 10)
    array_2d_view_assign(x[3::2, 3::3], 11)
    array_2d_view_assign(x[1:2, 2:3], 12)
    array_1d_view_assign(x[0, :], 1)
    array_1d_view_assign(x[1, ::2], 2)
    array_1d_view_assign(x[2, 1:4:2], 3)
    array_1d_view_assign(x[3, 3:4], 4)
    array_1d_view_assign(x[:, 0], 5)
    array_1d_view_assign(x[::2, 1], 6)
    array_1d_view_assign(x[1:4:2, 2], 7)
    array_1d_view_assign(x[3:4, 3], 8)
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            print(x[i][j])

def array_3d_view():
    x = np.zeros((3, 4, 5), dtype=int)
    array_3d_view_assign(x[::, ::, ::], 7)
    array_3d_view_assign(x[:2, :2:2, :2:3], 8)
    array_3d_view_assign(x[3::2, 3:4:3, :2:4], 9)
    array_2d_view_assign(x[:, :, 0], 4)
    array_2d_view_assign(x[::2, 1, 2:5:3], 5)
    array_2d_view_assign(x[2, 1::2, 2:], 6)
    array_1d_view_assign(x[0, 1, :], 1)
    array_1d_view_assign(x[1, ::3, 2], 2)
    array_1d_view_assign(x[1:2, 3, 4], 3)
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            for k in range(np.shape(x)[2]):
                print(x[i][j][k])

if __name__ == '__main__':
    array_view()
    array_view_2()
    array_view_3()
    array_view_negative_literal_step()
    array_view_negative_literal_step__2()
    array_view_negative_literal_step__3()
    array_view_negative_variable_step()
    array_view_negative_var()
    array_view_positive_literal()
    array_view_negative_literal()
    array_1d_view()
    array_2d_view()
    array_3d_view()
