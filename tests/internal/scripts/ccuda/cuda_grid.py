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
