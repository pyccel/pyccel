from pyccel.decorators import kernel, types
from pyccel import cuda
import cupy as cp
import math

@kernel
def func(arr:'int[:]'):
    i = cuda.grid(0)
    arr[i] = math.pow(arr[i], 2)

if __name__ == '__main__':
    a = cuda.array([1,2,3,4], memory_location='device')
    b = cuda.array([3,4,5,6], memory_location='host')
    c = b
    print(b)
    func[1,4](a)
    cuda.synchronize()
