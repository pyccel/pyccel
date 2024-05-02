# pylint: disable=missing-function-docstring, missing-module-docstring

def fib_caller(x : int):
    def fib(n : int) -> int:
        if n < 2:
            return n
        i = fib(n-1)
        j = fib(n-2)
        return i + j

    m = fib(x)
    return m
