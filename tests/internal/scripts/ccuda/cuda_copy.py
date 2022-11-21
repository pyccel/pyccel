from pyccel.decorators import kernel, types
from pyccel import cuda

if __name__ == '__main__':
    arr = cuda.array([1,2,3,4], memory_location='host')
    cpy = cuda.copy(arr, 'device', is_async=True)