from pyccel.epyccel import epyccel
import numpy as np
from pyccel.decorators import cuda

@cuda
def launchkernel(mat_1:'int[:,:]', mat_2:'int[:,:]'):
    import numpy as np
    from pyccel.decorators           import types, cuda
    from pyccel.stdlib.internal.cuda import array, deviceSynchronize, threadIdx, blockDim, blockIdx
    @cuda
    @types('int[:,:]','int[:,:]', 'int[:,:]')
    def mat_prod(mat_p, mat_1d, mat_2d):
        i = blockIdx(0) * blockDim(0) + threadIdx(0)
        if (i < mat_p.shape[0] * mat_p.shape[1]):
            i_x = int(i / mat_p.shape[1])
            i_y = int(i % mat_p.shape[1])
            mat_p[i_x][i_y] = 0
            for j in range(mat_p.shape[0]):
                mat_p[i_x][i_y] += mat_1d[i_x][j] * mat_2d[j][i_y]

    mat_p = array(mat_1)
    mat_1d = array(mat_1)
    mat_2d = array(mat_2)
    mat_prod[3,3](mat_p, mat_1d, mat_2d)
    deviceSynchronize()

a = np.random.randint(-1000,1000, size=(1000, 1000))
b = np.random.randint(-1000,1000, size=(1000, 1000))

f = epyccel(launchkernel, language='ccuda')
f(a, b)
