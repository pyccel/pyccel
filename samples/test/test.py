# from numpy import array
from pyccel.decorators import kernel
from pyccel import cuda
from numpy import array as narray
# from pyccel.stdlib.internal import PyccelthreadIdx


@types('int[:]')
def abbah(z):
    return z[0]


@kernel
@types('int[:]')
def func(a):
    i = cuda.threadIdx(0)
    """
    test test
    """
    print("Hello World! ", a[i])
    a[i] += 1

if __name__ == "__main__":
    # b = narray([1, 2, 3], dtype='int', order='C')
    a = cuda.array([1, 2, 3], dtype='int', order='C')
    func[1, 3](a)
    cuda.synchronize()
    print(a)