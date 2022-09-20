from pyccel.decorators import kernel, types
from pyccel import cuda

@kernel
@types('int[:]')
def func(a):
    i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
    print("Hello World! ", a[i])
