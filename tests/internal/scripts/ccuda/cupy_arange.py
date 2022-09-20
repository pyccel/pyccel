from pyccel.decorators import kernel, types
from pyccel import cuda
import cupy as cp

@kernel
@types('int[:]')
def func(a):
    i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
    print("Hello World! ", a[i])

if __name__ == '__main__':
    threads_per_block = 32
    n_blocks = 1
    arr = cp.arange(32)
    cuda.deviceSynchronize()
    func[n_blocks, threads_per_block](arr)
    cuda.deviceSynchronize()