# Object has already been defined with type 'numpy.float32'. Type 'numpy.float64' in assignment is incompatible
# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

a = np.float32(4)
a += np.float64(5.0)
