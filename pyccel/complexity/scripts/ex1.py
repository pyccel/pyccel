# ...
def add( x: 'int', y: 'int' ):
    return x+y

# ...
def sub( x: 'int', y: 'int' ):
    return x-y

# ...
def mul( x: 'int', y: 'int' ):
    return x*y

# ...
def div( x: 'int', y: 'int' ):
    return x/y

# ...
def iadd( x: 'int', y: 'int' ):
    x += y

# ...
def isub( x: 'int', y: 'int' ):
    x -= y

# ...
def imul( x: 'int', y: 'int' ):
    x *= y

# ...
def idiv( x: 'int', y: 'int' ):
    x /= y

# ...
def sum_natural_numbers( n: 'int' ):
    x = 0
    for i in range( 1, n+1 ):
        x += i
    return x

# ...
def factorial( n: 'int' ):
    x = 1
    for i in range( 2, n+1 ):
        x *= i
    return x

# ...
def fibonacci( n: 'int' ):
    x = 0
    y = 1
    for i in range( n ): # pylint: disable=unused-variable
        z = x+y
        x = y
        y = z
    return x

# ...
def double_loop( n: 'int' ):
    x = 0
    for i in range( 3, 10 ): # pylint: disable=unused-variable
        x += 1
        y  = n*x
        for j in range( 4, 15 ): # pylint: disable=unused-variable
            z = x-y
    return z

# ...
def double_loop_on_2d_array_C( z: 'int[:,:](order=C)' ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j

# ...
def double_loop_on_2d_array_F( z: 'int[:,:](order=F)' ):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j

# ...
def array_int32_1d_scalar_add( x: 'int32[:]', a: 'int32' ):
    x[:] += a

# ...
def array_int32_2d_scalar_add( x: 'int32[:,:]', a: 'int32' ):
    x[:,:] += a

