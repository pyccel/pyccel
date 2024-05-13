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
    for i in range( n ):
        z = x+y
        x = y
        y = z
    return x

# ...
def double_loop(n : int):
    x = 0
    for i in range( 3, 10 ):
        x += 1
        y  = n*x
        for j in range( 4, 15 ):
            z = x-y
    return z

# ...
def sum_natural_numbers_2(n : int):
    x = 0.
    for i in range( 1, n+1 ):
        x= x+i
    return x

# ...
def factorial_2(n : int):
    x = 1.
    for i in range( 2, n+1 ):
        x = x*i
    return x

# ...
def fibonacci_2(n : int):
    x = 0.
    y = 1.
    for i in range( n ):
        z = x+y
        x = y
        y = z
    return x

# ...
def double_loop_2(n : int):
    x = 0.
    for i in range( 3, 10 ):
        x = x+1
        y  = n*x
        for j in range( 4, 15 ):
            z = x-y
    return z

