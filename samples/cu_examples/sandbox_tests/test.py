from pyccel.epyccel import epyccel
from pyccel.decorators import types
from timeit import timeit
import numpy as np

@types('int', results='int')
def Fibonacci(n):
    if n<0:
        print("Incorrect input")
    # First Fibonacci number is 0
    elif n==0:
        return 0
    # Second Fibonacci number is 1
    elif n==1:
        return 1
    else:
        return Fibonacci(n-1)+Fibonacci(n-2)

g = epyccel(Fibonacci, language='c')
f = g(42)
# z = Fibonacci(35)
print(f)