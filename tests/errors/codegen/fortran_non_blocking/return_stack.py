# Can't return a stack array of unknown size
# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel.decorators import stack_array

@stack_array('a')
def build_a_boy(n : int):
    a = np.ones(n)
    return a

