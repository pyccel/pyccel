# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring
# pylint: disable=django-not-configure

from pyccel.decorators import kernel, types
from pyccel import cuda

@kernel
@types('int[:]')
def func_1d(a):
    i = cuda.grid(0)
    print("1 dim :", a[i])

@kernel
@types('int[:]')
def func_2d(a):
    i, j = cuda.grid(1)
    print("2 dim :", a[i], a[j])

@kernel
@types('int[:]')
def func_3d(a):
    i, j, k = cuda.grid(2)
    print("3 dim :", a[i], a[j], a[k])

if __name__ == '__main__':
    threads_per_block = 5
    n_blocks = 1
    arr = cuda.array([0, 1, 2, 3, 4])
    cuda.synchronize()
    func_1d[n_blocks, threads_per_block](arr)
    # Since we dont support multi-dim n_block / threads_per_block
    # func_2d and func_3d won't compile
    # func_2d[n_blocks, threads_per_block](arr)
    # func_3d[n_blocks, threads_per_block](arr)
    cuda.synchronize()
