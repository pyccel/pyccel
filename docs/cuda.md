# Getting started GPU

Pyccel now supports NVIDIA CUDA, empowering users to accelerate numerical computations on GPUs seamlessly. With Pyccel's high-level syntax and automatic code generation, harnessing the power of CUDA becomes effortless. This documentation provides a quick guide to enabling CUDA in Pyccel

## Cuda Decorator

### kernel

The kernel decorator allows the user to declare a CUDA kernel. The kernel can be defined in Python, and the syntax is similar to that of Numba.

```python
from pyccel.decorators import kernel

@kernel
def my_kernel():
    pass

blockspergrid = 1
threadsperblock = 1
# Call your kernel function
my_kernel[blockspergrid, threadsperblock]()

```

### device

Device functions are similar to kernels, but are executed within the context of a kernel. They can be called only from kernels or device functions, and are typically used for operations that are too small to justify launching a separate kernel, or for operations that need to be performed repeatedly within the context of a kernel.

```python
from pyccel.decorators import device, kernel

@device
def add(x, y):
    return x + y

@kernel
def my_kernel():
    x = 1
    y = 2
    z = add(x, y)
    print(z)

my_kernel[1, 1]()

```

## Cuda Arrays

Pyccel provides support for CUDA arrays, enabling efficient data transfer between the host and the GPU device. Here are some of the key functions you can use:

### cuda+host_empty

The cuda+host_empty function allocates an empty array on the host.

```python
from  pyccel import cuda

a = cuda.host_empty(10, 'int')

for i in range(10):
    a[i] = 1

if __name__ == '__main__':
    print(a)
```






