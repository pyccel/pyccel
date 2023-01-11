# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring

import cupy as cp
from pyccel import cuda

if __name__ == '__main__':
    c = ((1, 2), (1, 3))
    host_arr = cuda.array(c, dtype=int)
    device_arr = cuda.array(c, dtype=int, memory_location='device')
    arr1 = cp.ravel(host_arr)
    arr2 = cp.ravel(device_arr)
    arr3 = cp.ravel(c)
