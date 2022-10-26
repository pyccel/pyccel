from pyccel.decorators import kernel, types
from pyccel import cuda
import cupy as cp

@kernel
@types('int[:]')
def func(a):
    i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
    print("Hello World! ", a[i])

if __name__ == '__main__':
    threads_per_block = 5
    n_blocks = 1
    cp_arr = cp.array([[0, 1], [2, 3], [4, 2]])
    flat_cp_arr = cp.array([0, 1, 2, 3, 4])
    
    arr1 = cp.ravel(cp_arr)
    arr2 = cp.ravel(flat_cp_arr)
    
    arr2 = cp.ravel([0, 1, 2, 3, 4])
    arr3 = cp.ravel([[0, 1], [2, 3], [4, 2]])
    
    cuda.deviceSynchronize()
    func[n_blocks, threads_per_block](arr)
    cuda.deviceSynchronize()