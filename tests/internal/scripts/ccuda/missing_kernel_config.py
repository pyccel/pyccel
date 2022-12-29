# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import kernel, types
from pyccel import cuda

@kernel
@types('int[:]', 'int[:]')
def opp(a, b):
    index = cuda.grid(0)
    a[index] = a[index] - b[index]

if __name__ == '__main__':
    a = cuda.array([0,1,2,3,4], memory_location = 'device')
    b = cuda.array([1,2,3,4,5], memory_location = 'device')
    cuda.synchronize()
    #call kernel launch without specifying config
    opp(a, b)
    cuda.synchronize()
    h_a = cuda.copy(a, memory_location='host')

