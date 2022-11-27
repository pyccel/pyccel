import math
from numba import cuda

@cuda.jit
def addmul(x, y, out):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i >= x.shape[0]:
        return
    out[i] = x[i] + y[i] * math.fabs(x[i])

def numba_wrap_addmul(nb, tpb, size, x1, x2):
    d_x1 = cuda.to_device(x1)
    d_x2 = cuda.to_device(x2)
    d_out = cuda.device_array(size)
    cuda.synchronize()
    addmul[nb,128](d_x1, d_x2, d_out)
    cuda.synchronize()
    out = d_out.copy_to_host()
    cuda.synchronize()
    return out