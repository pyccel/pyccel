# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np
from pyccel.decorators import allow_negative_index

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

array_view()
array_view_negative_literal_step()
array_view_negative_literal_step__2()
array_view_negative_literal_step__3()
array_view_negative_variable_step()
