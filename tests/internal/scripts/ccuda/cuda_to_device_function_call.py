from pyccel.decorators import kernel, types
from pyccel import cuda
import numpy as np

@kernel
@types('int[:]', 'int[:]')
def mult(a, b):
    index = cuda.blockIdx(0) * cuda.blockDim(0) + cuda.threadIdx(0)
    a[index] = b[index] * a[index]

if __name__ == '__main__':
    threads_per_block = 5
    n_blocks = 1
    a = cuda.to_device(np.array([4, 3, 2, 1, 0]))
    b = cuda.to_device(np.array([1, 2, 3, 4, 5]))
    cuda.deviceSynchronize()
    mult[n_blocks, threads_per_block](a, b)
    cuda.deviceSynchronize()
    