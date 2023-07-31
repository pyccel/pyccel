# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel.decorators import types

@types('int', 'int')
def my_pow(n, m):
    return n ** m

def array_func_mult():
    arr = np.array([1,2,3,4])
    arr1 = arr * my_pow(2, 3)
    shape = np.shape(arr1)
    return arr[0], arr1[0], len(shape), shape[0]

def array_func_div():
    arr = np.array([1,2,3,4])
    arr1 = arr / my_pow(2, 3)
    shape = np.shape(arr1)
    return arr[0], arr1[0], len(shape), shape[0]

def array_arithmetic_op_func_call_1():
    arr = np.array([1,2,3,4])
    arr1 = np.array(arr * 2)
    shape = np.shape(arr1)
    return arr[0], arr1[0], len(shape), shape[0]

def array_arithmetic_op_func_call_2():
    arr = np.array([1,2,3,4])
    arr1 = np.array(arr / 2)
    shape = np.shape(arr1)
    return arr[0], arr1[0], len(shape), shape[0]

def array_arithmetic_op_func_call_3():
    arr = np.array([1,2,3,4])
    arr1 = np.array(arr * my_pow(2, 2))
    shape = np.shape(arr1)
    return arr[0], arr1[0], len(shape), shape[0]

def array_arithmetic_op_func_call_4():
    arr = np.array([1,2,3,4])
    arr1 = np.array(arr / my_pow(2, 2) + arr * 2)
    shape = np.shape(arr1)
    return arr[0], arr1[0], len(shape), shape[0]

def array_arithmetic_op_func_call_5():
    arr = np.array([1,2,3,4])
    arr1 = np.where(arr > 5, arr, (arr * 2) + arr)
    shape = np.shape(arr1)
    return arr[0], arr1[0], len(shape), shape[0]

def array_arithmetic_op_func_call_6():
    arr = np.array([1,2,3,4])
    arr1 = np.where(arr < 5, arr / 2, arr * 2)
    shape = np.shape(arr1)
    return arr[0], arr1[0], len(shape), shape[0]

def array_arithmetic_op_func_call_7():
    arr = np.array([1,2,3,4])
    arr1 = np.array([4,3,2,1])
    arr2 = np.array(arr + arr1)
    shape = np.shape(arr2)
    return arr[0], arr2[0], len(shape), shape[0]

def array_arithmetic_op_func_call_8():
    arr = np.array([1,2,3,4])
    arr1 = np.array([4,3,2,1])
    arr2 = np.array(arr - arr1)
    shape = np.shape(arr2)
    return arr[0], arr2[0], len(shape), shape[0]


if __name__ == "__main__":
    a_0, a1_0, ls_0, s_0 = array_func_mult()
    print(a_0, a1_0, ls_0, s_0)
    a_1, a1_1, ls_1, s_1 = array_func_div()
    print(a_1, a1_1, ls_1, s_1)
    a_2, a1_2, ls_2, s_2 = array_arithmetic_op_func_call_1()
    print(a_2, a1_2, ls_2, s_2)
    a_3, a1_3, ls_3, s_3 = array_arithmetic_op_func_call_2()
    print(a_3, a1_3, ls_3, s_3)
    a_4, a1_4, ls_4, s_4 = array_arithmetic_op_func_call_3()
    print(a_4, a1_4, ls_4, s_4)
    a_5, a1_5, ls_5, s_5 = array_arithmetic_op_func_call_4()
    print(a_5, a1_5, ls_5, s_5)
    a_6, a1_6, ls_6, s_6 = array_arithmetic_op_func_call_5()
    print(a_6, a1_6, ls_6, s_6)
    a_7, a1_7, ls_7, s_7 = array_arithmetic_op_func_call_6()
    print(a_7, a1_7, ls_7, s_7)
    a_8, a1_8, ls_8, s_8 = array_arithmetic_op_func_call_7()
    print(a_8, a1_8, ls_8, s_8)
    a_9, a1_9, ls_9, s_9 = array_arithmetic_op_func_call_8()
    print(a_9, a1_9, ls_9, s_9)
