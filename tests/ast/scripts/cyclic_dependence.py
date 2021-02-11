
def fib_caller(x : int):
    @types('int', results='int')
    def fib(n):
        if n < 2:
            return n
        i = fib(n-1)
        j = fib(n-2)
        return i + j

    m = fib(x)
    return m
