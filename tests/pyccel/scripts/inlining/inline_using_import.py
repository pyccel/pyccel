# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel.decorators import inline

@inline
def sin_2(d : float):
    return np.sin(2*d)
