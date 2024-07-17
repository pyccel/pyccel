# pylint: disable=missing-function-docstring, missing-module-docstring
from  pyccel import cuda

a = cuda.host_empty(10, 'int')

for i in range(10):
    a[i] = 1

if __name__ == '__main__':
    print(a)
