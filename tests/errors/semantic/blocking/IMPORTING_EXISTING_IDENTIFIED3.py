# pylint: disable=missing-function-docstring, missing-module-docstring/
# pylint: disable=reimported
import numpy as np
a = np.ones(3)

def f():
    import math as np
    x = np.sqrt(3)
    print(x)

b = np.zeros(3)


f()

import math as np
x = np.sqrt(3)
print(x)

import scipy.constants

print(scipy.constants.pi)
