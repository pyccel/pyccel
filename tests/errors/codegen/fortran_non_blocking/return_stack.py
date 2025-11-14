import numpy as np
from pyccel.decorators import stack_array

@stack_array('a')
def build_a_boy(n : int):
    a = np.ones(10)
    return a

