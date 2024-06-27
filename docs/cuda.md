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