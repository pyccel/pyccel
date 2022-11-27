from pyccel import cuda
from pyccel.decorators import kernel
import math

@kernel
def addmul(x: 'float64[:]', y: 'float64[:]', out: 'float64[:]'):
    i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)
    if i >= x.shape[0]:
        return
    out[i] = x[i] + y[i] * math.fabs(x[i])

def wrap_addmul(ng:'int', tn: 'int', x: 'float64[:]', y: 'float64[:]', z: 'float64[:]'):
    d_x = cuda.array(x, memory_location='device')
    d_y = cuda.array(y, memory_location='device')
    d_out = cuda.array(z, memory_location='device')
    cuda.deviceSynchronize()
    addmul[ng, tn](d_x, d_y, d_out)
    cuda.deviceSynchronize()
    out = cuda.copy(d_out, 'host')
    cuda.deviceSynchronize()
    return out