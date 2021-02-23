# from pyccel.stdlib.internal.cuda import array
from pyccel.decorators           import types
import numpy as np

@cuda
@types('int[:]','int[:]', 'int[:]')
def mat_prod(mat_1, mat_2, mat_p):
    x = 2
    x +=3

mat_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mat_2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

mat_p = np.array(mat_1)
# mat_prod[2,1](mat_1)

