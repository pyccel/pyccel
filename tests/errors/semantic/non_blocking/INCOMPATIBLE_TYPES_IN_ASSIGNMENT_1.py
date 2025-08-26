# Object has already been defined with type 'numpy.ndarray[numpy.int32]'. Type 'numpy.ndarray[numpy.int64]' in assignment is incompatible
# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

a = np.ones(4,dtype=np.int32)
a = np.zeros(4,dtype=np.int64)
