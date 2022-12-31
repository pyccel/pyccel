from pyccel.decorators import kernel, types
from pyccel import cuda

@kernel
@types('int[:]')
def func(a):
    i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
    print("Hello World! ", a[i])

if __name__ == '__main__':
    threads_per_block = 5
    n_blocks = 1
    arr = cuda.array([0, 1, 2, 3, 4])
    cuda.synchronize()
    func[n_blocks, threads_per_block](arr)
    cuda.synchronize()