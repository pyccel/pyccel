# Expecting an argument of the same type as the elements of the list
# pylint: disable=missing-function-docstring, missing-module-docstring

import numpy as np

a = [1,2,3]
b = np.int16(4)
a.insert(3, b)
