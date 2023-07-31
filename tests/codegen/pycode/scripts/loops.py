# pylint: disable=missing-function-docstring, missing-module-docstring

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
def double_loop_on_2d_array_C(z : 'int[:):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j


# ...
def double_loop_on_2d_array_F(z : 'int[:):

    from numpy import shape

    m, n = shape( z )

    for i in range( m ):
        for j in range( n ):
            z[i,j] = i-j

# ...
def product_loop_on_real_array(z : 'float[:], float[:]'):

    from numpy     import shape

    n, = shape( z )

    for i in range(n):
        out[i] = z[i]**2
