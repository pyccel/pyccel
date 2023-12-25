 # pylint: disable=missing-function-docstring, missing-module-docstring

#$ header function function(int)
def function(a):
    return a

#$ header function f1(int)
def f1(a):
    return a

#$ header function f2(int)
def f2(a):
    return a * 2

#$ header function f3(int)
def f3(a):
    return a * 5

#$ header function f4(int, float)
def f4(a, b):
    return a + b

#$ header function f5(float, float, float)
def f5(a, b, c):
    return a * b + c

#$ header function f6(int, int)
def f6(a, b):
    return a * 5 + b

#$ header function f7(float, float)
def f7(a, b):
    return a * 5 + b

#$ header function f8()
def f8():
    return 0.5

#$ header function f9()
def f9():
    return 0.5, 0.3

#$ header function high_int_1((int)(int), int)
def high_int_1(function, a):
    x = function(a)
    return x

#$ header function high_int_int_1((int)(int), (int)(int), int)
def high_int_int_1(function1, function2, a):
    x = function1(a)
    y = function2(a)
    return x + y

#$ header function high_float_1((float)(int, float), int, float)
def high_float_1(function, a, b):
    x = function(a, b)
    return x

#$ header function high_float_2((float)(float, float), float, float)
def high_float_2(function, a, b):
    x = function(a, b)
    return x

#$ header function high_float_3((float)())
def high_float_3(function):
    x = function()
    return x

#$ header function high_valuedarg_1(int, (int)(int))
def high_valuedarg_1(a, function=f1):
    x = function(a)
    return x

#$ header function high_float_float_int_1((float)(float, float), (float)(int, float), (int)(int))
def high_float_float_int_1(func1, func2, func3):
    x = func1(1.1, 11.2) + func2(11, 10.2) + func3(10)
    return x

def high_float_4(function : '(float)()'):
    x = function()
    return x

#$ header function high_float_5((float,float)())
def high_float_5(function):
    x,y = function()
    return x+y

def high_valuedarg_2(a : 'int', function : '(int)(int)' = f1):
    x = function(a)
    return x

def high_float_float_int_2(func1 : '(float)(float, float)', func2 : '(float)(int, float)', func3 : '(int)(int)'):
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

#$ header function test_float_1()
def test_float_1():
    x = high_float_1(f4, 10, 10.5)
    return x

#$ header function test_float_2()
def test_float_2():
    x = high_float_2(f7, 999.11, 10.5)
    return x

#$ header function test_float_3()
def test_float_3():
    x = high_float_3(f8)
    return x

#$ header function test_valuedarg_1()
def test_valuedarg_1():
    x = high_valuedarg_1(2)
    return x

#$ header function test_float_float_int_1()
def test_float_float_int_1():
    x = high_float_float_int_1(f7, f4, f3)
    return x

def test_float_4():
    x = high_float_4(f8)
    return x

#$ header function test_float_5()
def test_float_5():
    x = high_float_5(f9)
    return x

def test_valuedarg_2():
    x = high_valuedarg_2(2)
    return x

def test_float_float_int_2():
    x = high_float_float_int_2(f7, f4, f3)
    return x

def euler (dydt: '()(float, const float[:], float[:])',
           t0: 'float', t1: 'float', y0: 'float[:]', n: int,
           t: 'float[:]', y: 'float[:,:]'):

    dt = ( t1 - t0 ) / float ( n )
    y[0] = y0[:]

    for i in range ( n ):
        dydt ( t[i], y[i,:], y[i+1,:] )
        y[i+1,:] = y[i,:] + dt * y[i+1,:]

def predator_prey_deriv ( t: 'float', rf: 'float[:]', out: 'float[:]' ):

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

    t = [t0+(t1-t0)*i/n for i in range(n+1)]
    t = linspace ( t0, t1, n + 1 )
    y = zeros ( ( n + 1, m ) )

    euler ( predator_prey_deriv, t0, t1, y0, n, t, y )

    y0[:] = y[-1,:]
