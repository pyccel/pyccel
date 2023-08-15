# pylint: disable=missing-function-docstring, missing-module-docstring

def add_mixed_order(a : 'float[:,:]', b : 'float[:,:](order=F)'):
    a[:] = a + b

def mul_mixed_order(a : 'float[:,:]', b : 'float[:,:](order=F)'):
    a[:] = a * b

def sub_mixed_order(a : 'float[:,:]', b : 'float[:,:](order=F)'):
    a[:] = a - b

def div_mixed_order(a : 'float[:,:]', b : 'float[:,:](order=F)'):
    a[:] = a / b

def augadd_mixed_order(a : 'float[:,:]', b : 'float[:,:](order=F)'):
    a[:] += b

def augmul_mixed_order(a : 'float[:,:]', b : 'float[:,:](order=F)'):
    a[:] *= b

def augsub_mixed_order(a : 'float[:,:]', b : 'float[:,:](order=F)'):
    a[:] -= b

def augdiv_mixed_order(a : 'float[:,:]', b : 'float[:,:](order=F)'):
    a[:] /= b

def mul_by_vector_C(a : 'int[:,:]', b : 'int[:]'):
    a[:] *= b

def mul_by_vector_F(a : 'int[:,:](order=F)', b : 'int[:]'):
    a[:] *= b

def mul_by_vector_dim_1_C_C(a : 'int[:,:]'):
    import numpy as np
    b = np.array([[1],[2],[3]])
    a[:] *= b

def mul_by_vector_dim_1_C_F(a : 'int[:,:]'):
    import numpy as np
    b = np.array([[1],[2],[3]], order = 'F')
    a[:] *= b

def mul_by_vector_dim_1_F_C(a : 'int[:,:](order=F)'):
    import numpy as np
    b = np.array([[1],[2],[3]])
    a[:] *= b

def mul_by_vector_dim_1_F_F(a : 'int[:,:](order=F)'):
    import numpy as np
    b = np.array([[1],[2],[3]], order = 'F')
    a[:] *= b

def multi_dim_sum(result : 'int[:,:,:]', a : 'int[:,:,:]', b : 'int[:,:]', c : 'int[:]', d : 'int'):
    result[:,:,:] = a + b + c + d

def multi_dim_sum_ones(result : 'int[:,:,:]', a : 'int[:,:,:]'):
    import numpy as np
    s = np.shape(a)
    b = np.empty((1,s[1],s[2]),dtype=int)
    c = np.empty((1,   1,s[2]),dtype=int)
    b[:,:,:] = a[0]
    c[:,:,:] = a[0,0]
    d = a[0,0,0]
    result[:,:,:] = a + b + c + d

def multi_expression_assign(a : 'int[:,:]', b : 'int[:]'):
    import numpy as np
    a[:] = a * b
    a[:] = a * 2
    b[:] = b + 4
    a[:] = a - b
    a += np.sum(b)

def multi_expression_augassign(a : 'int[:,:]', b : 'int[:]'):
    import numpy as np
    a[:] *= b
    a[:] *= 2
    b[:] += 4
    a[:] -= b
    a += np.sum(b)

def grouped_expressions(a : 'int[:,:](order=F)', b : 'int[:,:]', c : 'int[:]'):
    import numpy as np
    a[:] = a - c
    a[:] = a * b
    a[:] = a + b
    a[:] = a + c
    a += np.sum(b)

def grouped_expressions2(a : 'int[:,:,:]', b : 'int[:,:]', c : 'int[:]'):
    import numpy as np
    a[:] = a - c
    a[:] = a * b
    a[:] = a + b
    a[:] = a + c
    a += np.sum(b)

def dependencies(a : 'int[:,:]', b : 'int[:]'):
    import numpy as np

    c = np.zeros_like(a)

    a[:] += b
    c += b*np.sum(a)

    a[:] = c[:]

def auto_dependencies(a : 'int[:,:]', b : 'int[:]'):
    import numpy as np

    c = np.ones_like(a)

    a[:] += b
    c += b*np.sum(c)

    a[:] = c[:]
