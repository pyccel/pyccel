# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring/
#
## This test should be moved to return_numpy_arrays.py once get fixed
#

import numpy as np

def single_return_rank():
    x = np.array([1,2,3,4])
    return x

if __name__ == '__main__':
    a = single_return_rank() + 1
    print(a)
