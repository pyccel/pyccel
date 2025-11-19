# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel.decorators import allow_negative_index

#==============================================================================

def sum_natural_numbers(n : int):
    x = 0
    for i in range( 1, n+1 ):
        x += i
    return x

# ...
def factorial(n : int):
    x = 1
    for i in range( 2, n+1 ):
        x *= i
    return x

# ...
def fibonacci(n : int):
    x = 0
    y = 1
    for i in range( n ): # pylint: disable=unused-variable
        z = x+y
        x = y
        y = z
    return x

# ...
def double_loop(n : int):
    x = 0
    for i in range( 3, 10 ): # pylint: disable=unused-variable
        x += 1
        y  = n*x
        for j in range( 4, 15 ): # pylint: disable=unused-variable
            z = x-y
    return z

# ...
def double_loop_on_2d_array_C(z : 'int[:,:](order=C)'):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j


# ...
def double_loop_on_2d_array_F(z : 'int[:,:](order=F)'):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j

# ...
def product_loop_on_2d_array_C(z : 'int[:,:](order=C)'):

    from numpy     import shape
    from itertools import product

    m, n = shape( z )

    x = [i for i in range(m)]
    y = [j for j in range(n)]

    for i,j in product( x, y ):
        z[i,j] = i-j

# ...
def product_loop_on_2d_array_F(z : 'int[:,:](order=F)'):

    from numpy     import shape
    from itertools import product

    m, n = shape( z )

    x = [i for i in range(m)]
    y = [j for j in range(n)]

    for i,j in product( x, y ):
        z[i,j] = i-j

# ...
def product_loop( z : 'float[:]', m : int, n : int ):

    from itertools import product

    x = [i*3+2 for i in range(m)]
    y = [j*7+6 for j in range(n)]

    k = 0
    for i,j in product( x, y ):
        z[k] = i-j
        k += 1

# ...
def map_on_1d_array(z : 'int[:]'):

    def f(x : int):
        return x+5

    res = 0
    for v in map( f, z ):
        res *= v

    return res

# ...
def enumerate_on_1d_array(z : 'int[:]'):

    res = 0
    for i,v in enumerate( z ):
        res += v*i

    return res

# ...
def enumerate_on_1d_array_with_start(z : 'int[:]', k : 'int'):

    res = 0
    for i,v in enumerate( z, k ):
        res += v*i

    return res

# ...
def enumerate_on_1d_array_with_tuple(z : 'int[:]'):

    res = 0
    for v in enumerate( z ):
        res += v[0]*v[1]

    return res

# ...
def zip_prod(m : int):

    x = [  i for i in range(m)]
    y = [2*j for j in range(m)]

    res = 0
    for i1,i2 in zip( x, y ):
        res += i1*i2

    return res

# ...
def product_loop_on_real_array(z : 'float[:]', out : 'float[:]'):

    from numpy     import shape

    n, = shape( z )

    for i in range(n):
        out[i] = z[i]**2

# ...
def fizzbuzz_search_with_breaks(fizz : 'int', buzz : 'int', max_val : 'int'):
    for i in range(1,max_val+1):
        if i%fizz == 0 and i%buzz == 0:
            break
    return i

# ...
def fizzbuzz_sum_with_continue(fizz : int, buzz : int, max_val : int):
    fizzbuzz_sum = 0
    for i in range(1,max_val+1):
        if i%fizz != 0:
            continue
        if i%buzz != 0:
            continue
        fizzbuzz_sum += i
    return fizzbuzz_sum

# ...
def fibonacci_while(n : int):
    x = 0
    y = 1
    i = 1
    while i <= n:
        z = x+y
        x = y
        y = z
        i = i + 1
    return x

# ...
def sum_nat_numbers_while(n : int):
    x = 0
    i = 0
    while i <= n:
        x += i
        i = i + 1
    return x

# ...
def double_while_sum(n : int, m : int):
    x = 0
    y = 0
    i = 0
    while x <= n:
        while y <= m:
            i += y
            y = y + 1
        i += x
        x = x + 1
    return i

# ...
def factorial_while(n : int):
    x = 1
    i = 1
    while i <= n:
        x = i * x
        i = i + 1
    return x

def while_not_0(n : int):
    while n:
        n -= 1
    return n

def for_loop1(start : int, stop : int, step : int):
    x = 0
    for i in range(start, stop, step):
        x += i
    return x

def for_loop2():
    x = 0
    for i in range(1, 10, 1):
        x += i
    return x

def for_loop3():
    x = 0
    for i in range(10, 1, -2):
        x += i
    return x

def temp_array_in_loop(a : 'int[:]', b : 'int[:]'):
    c = np.zeros_like(a)
    d1 = np.zeros_like(a)
    d2 = np.zeros_like(a)
    for _ in range(1):
        for d in range(2):
            b[d] += d
        c[:] = b - a
        d1[:] = np.abs(c)
        d2[:] = np.abs(b - a)
    return d1, d2

def less_than_100(n : int):
    for i in range(100):
        if i > n:
            return True
    return False

def for_expression():
    c = np.array([1,2,3,4,5])
    d = np.array([2.0,3.0,1.0,9.0,11.0])
    sum_e = 0.0
    for ei in c+d:
        sum_e = ei

    return sum_e

def for_lists_of_lists():
    a = [[1,2], [3,4]]
    b = [[5,6], [7,8]]
    c = [0, 0]
    for ai, bi in zip(a, b):
        for i in range(2):
            c[i] = ai[i] + bi[i]
            bi[i] = -1

    return c[0], c[1], b[0][0], b[0][1], b[1][0], b[1][1]

@allow_negative_index('a')
def for_unknown_index_slice():
    start = 1
    end = 7
    a = np.arange(10)
    result = 0
    for ai in a[:end]:
        result += ai

    return result
