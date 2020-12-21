# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types('int[:,:]','int[:]')
def mul_by_vector_C(a,b):
    a[:] *= b

@types('int[:,:](order=F)','int[:]')
def mul_by_vector_F(a,b):
    a[:] *= b

@types('int[:,:]')
def mul_by_vector_dim_1_C_C(a):
    import numpy as np
    b = np.array([[1],[2],[3]])
    a[:] *= b

@types('int[:,:]')
def mul_by_vector_dim_1_C_F(a):
    import numpy as np
    b = np.array([[1],[2],[3]], order = 'F')
    a[:] *= b

@types('int[:,:](order=F)')
def mul_by_vector_dim_1_F_C(a):
    import numpy as np
    b = np.array([[1],[2],[3]])
    a[:] *= b

@types('int[:,:](order=F)')
def mul_by_vector_dim_1_F_F(a):
    import numpy as np
    b = np.array([[1],[2],[3]], order = 'F')
    a[:] *= b
