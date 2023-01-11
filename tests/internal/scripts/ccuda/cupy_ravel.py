# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring

import cupy as cp
from pyccel.decorators import kernel, types
from pyccel import cuda

if __name__ == '__main__':
    threads_per_block = 32
    n_blocks = 1
    arr1 = cp.ravel([[1,2],[1,3]])
    arr2 = cp.ravel([1,2,3,4])
