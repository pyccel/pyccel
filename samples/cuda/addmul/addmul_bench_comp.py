from mod_addmul_pyccel import wrap_addmul
import numpy as np
import cupy as cp
from pyccel import cuda
from pyccel.decorators import inline

@inline
def div_up(a: 'int', b: 'int'):
    return (((a) + (b)-1) // (b))

if __name__ == "__main__":
    size = 1000
    num_iter = 10000
    tpb = 256

    x1 = np.empty(size)
    x2 = np.empty(size)
    for i in range(size):
        x1[i] = cuda.random.uniform(0.0, 1.0)
        x2[i] = cuda.random.uniform(0.1, 1.0)
    out = np.empty(size)
    ng = div_up(size, tpb)
    for i in range(num_iter):
        a = wrap_addmul(ng, tpb, x1, x2, out)