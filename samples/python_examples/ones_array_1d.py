from pyccel.decorators import types
from numpy import ones

a = ones(4)

i = 0
while i < 4:
    print(a[i])
    i = i + 1