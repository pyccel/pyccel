import numpy as np
from numba import cuda

@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1

arr = np.arange(1000)
d_arr = cuda.to_device(arr)

increment_by_one[100, 100](d_arr)

result_array = d_arr.copy_to_host()
print(result_array)
