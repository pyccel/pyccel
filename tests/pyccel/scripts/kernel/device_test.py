# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import device, kernel
from pyccel import cuda

@device
def device_call():
    print("Hello from device")

@kernel
def kernel_call():
    device_call()

def f():
    kernel_call[1,1]()
    cuda.synchronize()

if __name__ == '__main__':
    f()
