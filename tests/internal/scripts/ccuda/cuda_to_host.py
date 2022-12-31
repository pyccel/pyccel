# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring/

import numpy as np
from pyccel.decorators import kernel, types
from pyccel import cuda

@kernel
@types('int[:]', 'int[:]')
def mult(a):
    index = cuda.blockIdx(0) * cuda.blockDim(0) + cuda.threadIdx(0)
    print(a[index])

if __name__ == '__main__':
    threads_per_block = 5
    n_blocks = 1
    a = np.array([0,1,2,3,4])
    b = cuda.to_device(a)
    c = cuda.to_host(b)
    print(c)
