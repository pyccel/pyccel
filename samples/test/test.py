# from numpy import array
from pyccel.decorators import kernel
from pyccel import cuda
from numpy import array as narray
# from pyccel.stdlib.internal import PyccelthreadIdx



@kernel
@types('int[:]')
def func(a):
    # i = cuda.threadIdx(0)
    """
    test test
    """
    print("Hello World!")

if __name__ == "__main__":
    b = narray([1, 2, 3], dtype='int', order='C')
    a = cuda.array([1, 2, 3], dtype='int', order='C')
    func[1, 2](a)