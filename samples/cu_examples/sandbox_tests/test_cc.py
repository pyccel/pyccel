from pyccel.stdlib.internal.cuda import array, deviceSynchronize, threadIdx, blockDim, blockIdx, cudaMalloc
from pyccel.decorators           import types
import numpy as np

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

mat_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mat_2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = cudaMalloc((3,), 'GPU', int)
mat_p = array(mat_1)
mat_1d = array(mat_1)
mat_2d = array(mat_2)
mat_prod[2,1](mat_p, mat_1d, mat_2d)
deviceSynchronize()
