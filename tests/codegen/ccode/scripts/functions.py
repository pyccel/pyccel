# pylint: disable=missing-function-docstring, missing-module-docstring

def fib(n : int) -> int:
    if n < 2:
        return n
    i = fib(n-1)
    j = fib(n-2)
    return i + j

def recu_func(x : int) -> int:
    if x > 0:
        x = x - 1
    return x

def helloworld():
    print('hello world')

def incr(x : int):
    x = x + 1

def decr(x : int) -> int:
    y = x - 1
    return y

def f1(x : int, n : int = 2, m : int = 3) -> int:
    y = x - n*m
    return y

def f2(x : int, m : int = None) -> int:
    if m is None:
        y = x + 1
    else:
        y = x - 1
    return y

y = decr(2)
z = f1(1)

z1 = f2(1)
z2 = f2(1, m=0)

helloworld()

print(y)
print(z)

print(z1)
print(z2)

def pass_fun():
    pass
