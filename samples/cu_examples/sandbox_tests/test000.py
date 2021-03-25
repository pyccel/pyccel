import cupy as cp
import numpy as np

# add_kernel = cp.RawKernel(r'''

# extern "C" __global__

# void my_add(const float* x1, const float* x2, float* y) {

#     int tid = blockDim.x * blockIdx.x + threadIdx.x;

#     y[tid] = x1[tid] + x2[tid];

# }
# ''', 'my_add')

x1 = cp.arange(25, dtype=np.float32)

x2 = cp.arange(25, dtype=np.float32)

y = cp.zeros((5, 5), dtype=np.float32)

# add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
