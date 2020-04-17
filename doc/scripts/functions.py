from pyccel.decorators import types

@types(int, int, int)
def f(x, n=2, m=3):
    y = x - n*m
    return y

x = f(5)
print(x)
