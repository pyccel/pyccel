# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types

@types('real[:,:]','real[:,:](order=F)')
def add_mixed_order(a,b):
    a[:] = a + b

@types('real[:,:]','real[:,:](order=F)')
def mul_mixed_order(a,b):
    a[:] = a * b

@types('real[:,:]','real[:,:](order=F)')
def sub_mixed_order(a,b):
    a[:] = a - b

@types('real[:,:]','real[:,:](order=F)')
def div_mixed_order(a,b):
    a[:] = a / b

@types('real[:,:]','real[:,:](order=F)')
def augadd_mixed_order(a,b):
    a[:] += b

@types('real[:,:]','real[:,:](order=F)')
def augmul_mixed_order(a,b):
    a[:] *= b

@types('real[:,:]','real[:,:](order=F)')
def augsub_mixed_order(a,b):
    a[:] -= b

@types('real[:,:]','real[:,:](order=F)')
def augdiv_mixed_order(a,b):
    a[:] /= b

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

@types('int[:,:,:]','int[:,:,:]','int[:,:]','int[:]','int')
def multi_dim_sum(result, a, b, c, d):
    result[:,:,:] = a + b + c + d

@types('int[:,:,:]','int[:,:,:]')
def multi_dim_sum_ones(result, a):
    import numpy as np
    s = np.shape(a)
    b = np.empty((1,s[1],s[2]),dtype=int)
    c = np.empty((1,   1,s[2]),dtype=int)
    b[:,:,:] = a[0]
    c[:,:,:] = a[0,0]
    d = a[0,0,0]
    result[:,:,:] = a + b + c + d

@types('int[:,:]','int[:]')
def multi_expression_assign(a,b):
    import numpy as np
    a[:] = a * b
    a[:] = a * 2
    b[:] = b + 4
    a[:] = a - b
    a += np.sum(b)

@types('int[:,:]','int[:]')
def multi_expression_augassign(a,b):
    import numpy as np
    a[:] *= b
    a[:] *= 2
    b[:] += 4
    a[:] -= b
    a += np.sum(b)

@types('int[:,:](order=F)','int[:,:]','int[:]')
def grouped_expressions(a,b,c):
    import numpy as np
    a[:] = a - c
    a[:] = a * b
    a[:] = a + b
    a[:] = a + c
    a += np.sum(b)

@types('int[:,:,:]','int[:,:]','int[:]')
def grouped_expressions2(a,b,c):
    import numpy as np
    a[:] = a - c
    a[:] = a * b
    a[:] = a + b
    a[:] = a + c
    a += np.sum(b)

@types('int[:,:]','int[:]')
def dependencies(a,b):
    import numpy as np

    c = np.zeros_like(a)

    a[:] += b
    c += b*np.sum(a)

    a[:] = c[:]

@types('int[:,:]','int[:]')
def auto_dependencies(a,b):
    import numpy as np

    c = np.ones_like(a)

    a[:] += b
    c += b*np.sum(c)

    a[:] = c[:]
