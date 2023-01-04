# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring

import math
from pyccel.decorators import kernel
from pyccel import cuda

@kernel
def func(arr:'int[:]'):
    """
    Takes an array 'arr' as input and raise each element to the power 2.
    """
    i = cuda.grid(0)
    arr[i] = math.pow(arr[i], 2)

if __name__ == '__main__':
    a = cuda.array([1,2,3,4], memory_location='device')
    func[1,4](a)
    c = a
    cuda.synchronize()
