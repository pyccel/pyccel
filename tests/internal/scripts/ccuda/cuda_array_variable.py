# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring/

from pyccel.decorators import kernel, types
from pyccel import cuda

@kernel
@types('int[:]')
def square(a):
    index = cuda.blockIdx(0) * cuda.blockDim(0) + cuda.threadIdx(0)
    a[index] = a[index] * a[index]

if __name__ == '__main__':
    threads_per_block = 5
    n_blocks = 1
    arr_var = [0,1,2,3,4]
    arr = cuda.array(arr_var)
    cuda.deviceSynchronize()
    square[n_blocks, threads_per_block](arr)
    cuda.deviceSynchronize()
