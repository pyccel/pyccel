# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import Final

def function(a : int):
    return a

def f1(a : int):
    return a

def f2(a : int):
    return a * 2

def f3(a : int):
    return a * 5

def f4(a : int, b : float):
    return a + b

def f5(a : float, b : float, c : float):
    return a * b + c

def f6(a : int, b : int):
    return a * 5 + b

def f7(a : float, b : float):
    return a * 5 + b

def f8():
    return 0.5

def f9():
    return 0.5, 0.3

def high_int_1(function : '(int)(int)', a : int):
    x = function(a)
    return x

def high_int_int_1(function1 : '(int)(int)', function2 : '(int)(int)', a : int):
    x = function1(a)
    y = function2(a)
    return x + y

def high_float_1(function : '(float)(int, float)', a : int, b : float):
    x = function(a, b)
    return x

def high_float_2(function : '(float)(float, float)', a : float, b : float):
    x = function(a, b)
    return x

def high_float_3(function : '(float)()'):
    x = function()
    return x

def high_valuedarg_1(a : int, function : '(int)(int)' = f1):
    x = function(a)
    return x

def high_float_float_int_1(func1 : '(float)(float, float)', func2 : '(float)(int, float)', func3 : '(int)(int)'):
    x = func1(1.1, 11.2) + func2(11, 10.2) + func3(10)
    return x

def high_float_4(function : '(float)()'):
    x = function()
    return x

def high_float_5(function : '(tuple[float,float])()'):
    x,y = function()
    return x+y

def high_valuedarg_2(a : 'int', function : '(int)(int)' = f1):
    x = function(a)
    return x

def high_float_float_int_2(func1 : '(float)(float, float)', func2 : '(float)(int, float)', func3 : '(int)(int)'):
    x = func1(1.1, 11.2) + func2(11, 10.2) + func3(10)
    return x

def test_int_1():
    x = high_int_1(f1, 0)
    return x

def test_int_int_1():
    x = high_int_int_1(f1, f2, 10)
    return x

def test_float_1():
    x = high_float_1(f4, 10, 10.5)
    return x

def test_float_2():
    x = high_float_2(f7, 999.11, 10.5)
    return x

def test_float_3():
    x = high_float_3(f8)
    return x

def test_valuedarg_1():
    x = high_valuedarg_1(2)
    return x

def test_float_float_int_1():
    x = high_float_float_int_1(f7, f4, f3)
    return x

def test_float_4():
    x = high_float_4(f8)
    return x

def test_float_5():
    x = high_float_5(f9)
    return x

def test_valuedarg_2():
    x = high_valuedarg_2(2)
    return x

def test_float_float_int_2():
    x = high_float_float_int_2(f7, f4, f3)
    return x

def euler (dydt: '()(float, Final[float[:]], float[:])',
           t0: 'float', t1: 'float', y0: 'float[:]', n: int,
           t: 'float[:]', y: 'float[:,:]'):

    dt = ( t1 - t0 ) / float ( n )
    y[0] = y0[:]

    for i in range ( n ):
        dydt ( t[i], y[i,:], y[i+1,:] )
        y[i+1,:] = y[i,:] + dt * y[i+1,:]

def predator_prey_deriv ( t: 'float', rf: 'Final[float[:]]', out: 'float[:]' ):

    r = rf[0]
    f = rf[1]

    drdt =    2.0 * r - 0.001 * r * f
    dfdt = - 10.0 * f + 0.002 * r * f

    out[0] = drdt
    out[1] = dfdt

def euler_test ( t0: 'float', t1 : 'float', y0: 'float[:]', n: int ):
    from numpy import zeros
    from numpy import linspace

    m = len ( y0 )

    t = linspace ( t0, t1, n + 1 )
    y = zeros ( ( n + 1, m ) )

    euler ( predator_prey_deriv, t0, t1, y0, n, t, y )

    y0[:] = y[-1,:]
