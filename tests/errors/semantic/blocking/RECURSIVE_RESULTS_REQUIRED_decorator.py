# pylint: disable=missing-function-docstring, missing-module-docstring

def fib(n : 'int') :
    if n < 2:
        result = n
        return result
    result = fib (n - 1) + fib (n - 2)
    return result

print(fib(20))
