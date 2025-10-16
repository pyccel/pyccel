# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import numpy as np
from pyccel.decorators import inline

class UnusedInline:
    @inline
    def sin_2(self, d : float):
        return np.sin(2 * d)
