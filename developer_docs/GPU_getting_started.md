# Getting started GPU

## Decorators:

* Kernel

    In Pyccel, the `@kernel` decorator is used to indicate that a function should be treated as a CUDA kernel. A CUDA kernel is a function that is executed on a GPU and is typically used to perform a parallel computation over a large dataset.

    The `@kernel` decorator is used to indicate to Pyccel that the function should be compiled as a CUDA kernel, and that it can be launched on a GPU using the appropriate syntax. For example, if we have a function decorated with `@kernel` like this:

    > kernels can not return a variable that is why we pass the returned variable in the kernel arguments.

    ```Python
    from pyccel import cuda
    from pyccel.decorators import kernel

    @kernel
    def my_kernel(x: 'float64[:]', y: 'float64[:]', out: 'float64[:]'):
        i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
        if i >= x.shape[0]:
            return
        out[i] = x[i] + y[i]
    ```
    We can launch this kernel on a GPU using the following syntax:
    ```Python
    ng = 128  # size of the grid
    tn = 64   # size of the block
    my_kernel[ng, tn](x, y, out)
    ```
* Device

    The `@device` decorator is similar to the `@kernel` decorator, but indicates that the function should be compiled and executed on the GPU as a device function, rather than a kernel.
    
    Device functions are similar to kernels, but are executed within the context of a kernel. They can be called only from kernels, and are typically used for operations that are too small to justify launching a separate kernel, or for operations that need to be performed repeatedly within the context of a kernel.

    ```Python
    from pyccel import cuda
    from pyccel.decorators import device

    @device
    def my_device_function(x: 'float32'):
        return x * x

    # Call the function from a kernel
    @kernel
    def my_kernel(x: 'float32[:]', y: 'float32[:]'):
        i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
        if i >= x.shape[0]:
            return
        y[i] = my_device_function(x[i])
    ```

## Built-in variables:

* cuda.threadIdx(dim): Returns the index of the current thread within the block in a given dimension dim. dim is an integer between 0 and the number of dimensions of the thread block minus one.

* cuda.blockIdx(dim): Returns the index of the current block within the grid in a given dimension dim.
dim is an integer between 0 and the number of dimensions of the grid minus one.

* cuda.blockDim(dim): Returns the size of the thread block in a given dimension dim. dim is an integer between 0 and the number of dimensions of the thread block minus one.

* cuda.gridDim(dim): Returns the size of the grid in a given dimension dim. dim is an integer between 0 and the number of dimensions of the grid minus one.

> These built-in variables are provided by the CUDA runtime and are available to all CUDA kernels, regardless of the programming language being used. They are used to obtain information about the execution configuration of the kernel, such as the indices and dimensions of the threads and blocks. They are an important part of the CUDA programming model and are often used to compute global indices and other information about the execution configuration of the kernel.

```Python
from pyccel import cuda
from pyccel.decorators import kernel

@kernel
def my_kernel(x: 'float64[:]', y: 'float64[:]', out: 'float64[:]'):
    i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
    if i >= x.shape[0]:
        return
    out[i] = x[i] + y[i]
```

The kernel uses the `cuda.threadIdx(0)`, `cuda.blockIdx(0)`, and `cuda.blockDim(0)` built-in variables to compute the global index of the current thread within the input arrays. 

The global index is computed as the sum of the thread index and the block index, multiplied by the block size, in the first dimension. This allows each thread to compute its own index within the input arrays, and to exit if its index falls outside the bounds of the arrays.
