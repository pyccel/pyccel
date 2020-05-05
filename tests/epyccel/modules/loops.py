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
    for i in range( n ):
        z = x+y
        x = y
        y = z
    return x

# ...
@types( int )
def double_loop( n ):
    x = 0
    for i in range( 3, 10 ):
        x += 1
        y  = n*x
        for j in range( 4, 15 ):
            z = x-y
    return z

# ...
@types( 'int[:,:](order=C)' )
def double_loop_on_2d_array_C( z ):

    from numpy import shape

    s = shape( z )
    m = s[0]
    n = s[1]

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j


# ...
@types( 'int[:,:](order=F)' )
def double_loop_on_2d_array_F( z ):

    from numpy import shape

    s = shape( z )
    m = s[0]
    n = s[1]

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j

# ...
@types( 'int[:,:](order=C)' )
def product_loop_on_2d_array_C( z ):

    from numpy     import shape
    from itertools import product

    s = shape( z )
    m = s[0]
    n = s[1]

    x = [i for i in range(m)]
    y = [j for j in range(n)]

    for i,j in product( x, y ):
        z[i,j] = i-j

# ...
@types( 'int[:,:](order=F)' )
def product_loop_on_2d_array_F( z ):

    from numpy     import shape
    from itertools import product

    s = shape( z )
    m = s[0]
    n = s[1]

    x = [i for i in range(m)]
    y = [j for j in range(n)]

    for i,j in product( x, y ):
        z[i,j] = i-j

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

    s = shape( z )
    n = s[0]

    for i in range(n):
        out[i] = z[i]**2
