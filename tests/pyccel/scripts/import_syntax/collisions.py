# pylint: disable=reimported
import numpy as np
a = np.ones(3)

def f():
    import math as np
    x = np.sqrt(3)
    print(x)

b = np.zeros(3)

np = a[0]+b[0]

print(np)

f()

import math as np
x = np.sqrt(3)
print(x)

import scipy.constants

print(scipy.constants.pi)
