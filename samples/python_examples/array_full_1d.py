from pyccel.decorators import types
from numpy import full

a = full(4, 5)

i = 0
while i < 4:
    print(a[i])
    i = i + 1