 # pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

#$ header function f1(int)
def f1(a):
    return a

#$ header function f2(int)
def f2(a):
    return a * 2

#$ header function f3(int)
def f3(a):
    return a * 5

#$ header function f4(int, real)
def f4(a, b):
    return a + b

#$ header function f5(real, real, real)
def f5(a, b, c):
    return a * b + c

#$ header function f6(int, int)
def f6(a, b):
    return a * 5 + b

#$ header function f7(real, real)
def f7(a, b):
    return a * 5 + b

#$ header function f8()
def f8():
    return 0.5

#$ header function f9(real, real) results(real, real)
def f9(x, y):
    return x * y, x / y

@types()
def f10():
    y = 0
    if y:
        pass

#$ header function high_int_1((int)(int), int)
def high_int_1(func, a):
    x = func(a)
    return x

#$ header function high_int_int_1((int)(int), (int)(int), int)
def high_int_int_1(function1, function2, a):
    x = function1(a)
    y = function2(a)
    return x + y

#$ header function high_real_1((real)(int, real), int, real)
def high_real_1(func, a, b):
    x = func(a, b)
    return x

#$ header function high_real_2((real)(real, real), real, real)
def high_real_2(func, a, b):
    x = func(a, b)
    return x

#$ header function high_real_3((real)())
def high_real_3(func):
    x = func()
    return x

#$ header function high_valuedarg_1(int, (int)(int))
def high_valuedarg_1(a, func=f1):
    x = func(a)
    return x

#$ header function high_real_real_int((real)(real, real), (real)(int, real), (int)(int))
def high_real_real_int(func1, func2, func3):
    x = func1(1.1, 11.2) + func2(11, 10.2) + func3(10)
    return x

#$ header function high_multi_real_1((real, real)(real, real), real, real)
def high_multi_real_1(func, x, y):
    x , y = func(x, y)
    return x + y

@types('()()')
def high_void_1(func):
    func()
    return 0
