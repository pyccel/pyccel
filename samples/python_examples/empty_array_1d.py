from pyccel.decorators import types
from numpy import empty

a = empty(4)

i = 0
while i < 4:
    print(a[i])
    i = i + 1