from pyccel.decorators import kernel
from pyccel import cuda


@kernel
def func():
    i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
    print("Hello World! ")

if __name__ == '__main__':
    func[1, 5]()
