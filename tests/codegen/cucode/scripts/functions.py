# pylint: disable=missing-function-docstring, missing-module-docstring

#$ header function fib(int) results(int)
def fib(n):
    if n < 2:
        return n
    i = fib(n-1)
    j = fib(n-2)
    return i + j

#$ header function recu_func(int) results(int)
def recu_func(x):
    if x > 0:
        x = x - 1
    return x

def helloworld():
    print('hello world')

#$ header function incr(int)
def incr(x):
    x = x + 1

#$ header function decr(int) results(int)
def decr(x):
    y = x - 1
    return y

#$ header function f1(int, int, int) results(int)
def f1(x, n=2, m=3):
    y = x - n*m
    return y

#$ header function f2(int, int) results(int)
def f2(x, m=None):
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
