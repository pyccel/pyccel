# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types

#==============================================================================

@types( int )
def sum_natural_numbers( n ):
    x = 0
    for i in range( 1, n+1 ):
        x += i
    return x

# ...
@types( int )
def factorial( n ):
    x = 1
    for i in range( 2, n+1 ):
        x *= i
    return x

# ...
@types( int )
def fibonacci( n ):
    x = 0
    y = 1
    for i in range( n ): # pylint: disable=unused-variable
        z = x+y
        x = y
        y = z
    return x

# ...
@types( int )
def double_loop( n ):
    x = 0
    for i in range( 3, 10 ): # pylint: disable=unused-variable
        x += 1
        y  = n*x
        for j in range( 4, 15 ): # pylint: disable=unused-variable
            z = x-y
    return z

# ...
@types( 'int[:,:](order=C)' )
def double_loop_on_2d_array_C( z ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j


# ...
@types( 'int[:,:](order=F)' )
def double_loop_on_2d_array_F( z ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j

# ...
@types( 'int[:,:](order=C)' )
def product_loop_on_2d_array_C( z ):

    from numpy     import shape
    from itertools import product

    m, n = shape( z )

    x = [i for i in range(m)]
    y = [j for j in range(n)]

    for i,j in product( x, y ):
        z[i,j] = i-j

# ...
@types( 'int[:,:](order=F)' )
def product_loop_on_2d_array_F( z ):

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
@types( 'int[:]' )
def map_on_1d_array( z ):

    @types( int )
    def f( x ):
        return x+5

    res = 0
    for v in map( f, z ):
        res *= v

    return res

# ...
@types( 'int[:]' )
def enumerate_on_1d_array( z ):

    res = 0
    for i,v in enumerate( z ):
        res += v*i

    return res

# ...
@types( 'int[:]', 'int' )
def enumerate_on_1d_array_with_start( z, k ):

    res = 0
    for i,v in enumerate( z, k ):
        res += v*i

    return res

# ...
@types( int )
def zip_prod( m ):

    x = [  i for i in range(m)]
    y = [2*j for j in range(m)]

    res = 0
    for i1,i2 in zip( x, y ):
        res += i1*i2

    return res

# ...
@types( 'real[:], real[:]' )
def product_loop_on_real_array( z, out ):

    from numpy     import shape

    n, = shape( z )

    for i in range(n):
        out[i] = z[i]**2

# ...
@types('int,int,int')
def fizzbuzz_search_with_breaks( fizz, buzz, max_val ):
    for i in range(1,max_val+1):
        if i%fizz == 0 and i%buzz == 0:
            break
    return i

# ...
@types('int,int,int')
def fizzbuzz_sum_with_continue( fizz, buzz, max_val ):
    fizzbuzz_sum = 0
    for i in range(1,max_val+1):
        if i%fizz != 0:
            continue
        if i%buzz != 0:
            continue
        fizzbuzz_sum += i
    return fizzbuzz_sum

# ...
@types(int)
def fibonacci_while(n):
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
@types(int)
def sum_nat_numbers_while(n):
    x = 0
    i = 0
    while i <= n:
        x += i
        i = i + 1
    return x

# ...
@types(int,int)
def double_while_sum(n, m):
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
@types( int )
def factorial_while( n ):
    x = 1
    i = 1
    while i <= n:
        x = i * x
        i = i + 1
    return x

@types( int )
def while_not_0( n ):
    while n:
        n -= 1
    return n

@types( int, int, int )
def for_loop1(start, stop, step):
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
    import numpy as np
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

