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

#$ header function high_int_1((int)(int), int)
def high_int_1(function, a):
    x = function(a)
    return x

#$ header function high_int_int_1((int)(int), (int)(int), int)
def high_int_int_1(function1, function2, a):
    x = function1(a)
    y = function2(a)
    return x + y

#$ header function high_real_1((real)(int, real), int, real)
def high_real_1(function, a, b):
    x = function(a, b)
    return x

#$ header function high_real_2((real)(real, real), real, real)
def high_real_2(function, a, b):
    x = function(a, b)
    return x

#$ header function high_real_3((real)())
def high_real_3(function):
    x = function()
    return x

#$ header function high_valuedarg_1(int, (int)(int))
def high_valuedarg_1(a, function=f1):
    x = function(a)
    return x

#$ header function high_real_real_int_1((real)(real, real), (real)(int, real), (int)(int))
def high_real_real_int_1(func1, func2, func3):
    x = func1(1.1, 11.2) + func2(11, 10.2) + func3(10)
    return x

@types('(real)()')
def high_real_4(function):
    x = function()
    return x

@types('int', '(int)(int)')
def high_valuedarg_2(a, function=f1):
    x = function(a)
    return x

@types('(real)(real, real)', '(real)(int, real)', '(int)(int)')
def high_real_real_int_2(func1, func2, func3):
    x = func1(1.1, 11.2) + func2(11, 10.2) + func3(10)
    return x

#$ header function test_int_1()
def test_int_1():
    x = high_int_1(f1, 0)
    return x

#$ header function test_int_int_1()
def test_int_int_1():
    x = high_int_int_1(f1, f2, 10)
    return x

#$ header function test_real_1()
def test_real_1():
    x = high_real_1(f4, 10, 10.5)
    return x

#$ header function test_real_2()
def test_real_2():
    x = high_real_2(f7, 999.11, 10.5)
    return x

#$ header function test_real_3()
def test_real_3():
    x = high_real_3(f8)
    return x

#$ header function test_valuedarg_1()
def test_valuedarg_1():
    x = high_valuedarg_1(2)
    return x

#$ header function test_real_real_int_1()
def test_real_real_int_1():
    x = high_real_real_int_1(f7, f4, f3)
    return x

@types()
def test_real_4():
    x = high_real_4(f8)
    return x

@types()
def test_valuedarg_2():
    x = high_valuedarg_2(2)
    return x

@types()
def test_real_real_int_2():
    x = high_real_real_int_2(f7, f4, f3)
    return x
