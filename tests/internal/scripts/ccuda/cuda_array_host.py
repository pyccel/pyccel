# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring/

from pyccel.decorators import kernel, types
from pyccel import cuda

@types('int[:]')
def multiplyPrintElements(a):
    for i in a:
        i = i * 2
        print(i)

if __name__ == '__main__':
     arr = cuda.array([0,1,2,3,4], memory_location='host')
     multiplyPrintElements(arr)
