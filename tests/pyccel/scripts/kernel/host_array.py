# pylint: disable=missing-function-docstring, missing-module-docstring
from  pyccel import cuda
def f():
    a = cuda.host_empty(10)

    for i in range(10):
        a[i] = 1
    print(a)
if __name__ == '__main__':
    f()
