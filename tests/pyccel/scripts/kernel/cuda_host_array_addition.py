# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel import cuda

def addition_cuda_host_arrays():
    a = cuda.host_empty(3)
    b = cuda.host_empty(3)

    for i in range(3):
        b[i] = 1
        a[i] = 1

    for i in range(3):
        b[i] += a[i]

    print(b)

if __name__ == '__main__':
    addition_cuda_host_arrays()
