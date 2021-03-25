import cupy
from pyccel.epyccel import epyccel
from pyccel.decorators import types, cuda

@cuda
@types('int[:,:]','int[:,:]', 'int[:,:]')
def mat_prod(mat_p, mat_1d, mat_2d):
    from pyccel.stdlib.internal.cuda import array, deviceSynchronize, threadIdx, blockDim, blockIdx, cudaMalloc
    i = blockIdx(0) * blockDim(0) + threadIdx(0)
    if (i < mat_p.shape[0] * mat_p.shape[1]):
        i_x = int(i / mat_p.shape[1])
        i_y = int(i % mat_p.shape[1])

        mat_p[i_x][i_y] = 0
        for j in range(mat_p.shape[0]):
            mat_p[i_x][i_y] += mat_1d[i_x][j] * mat_2d[j][i_y]

f = epyccel(mat_prod, language='ccuda')
