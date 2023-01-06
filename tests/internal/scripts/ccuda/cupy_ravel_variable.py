# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring

import cupy as cp
from pyccel.decorators import kernel, types
from pyccel import cuda

@kernel
@types('int[:]', 'int[:]', 'int[:]')
def func(a, b, c):
    i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
    print("Hello World! ", a[i], b[i], c[i])

if __name__ == '__main__':
    threads_per_block = 32
    n_blocks = 1
    c =[[1,2],[1,3]]
    host_arr = cuda.array(c, dtype=int)
    device_arr = cuda.array(c, dtype=int, memory_location='device')
    arr1 = cp.ravel(host_arr)
    arr2 = cp.ravel(device_arr)
    arr3 = cp.ravel(c)
    cuda.synchronize()
    func[n_blocks, threads_per_block](arr1, arr2, arr3)
    cuda.synchronize()
