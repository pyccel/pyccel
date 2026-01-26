# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np


# +=

def augassign_add_1d_int(a : 'int[:]'):
    b = a
    b += 42
    return b[0]

def augassign_add_1d_float(a : 'float[:]'):
    b = a
    b += 4.2
    return b[0]

def augassign_add_1d_complex(a : 'complex[:]'):
    b = a
    b += (4.0 + 2.0j)
    return b[0]

def augassign_add_2d_int(a : 'int[:,:]'):
    b = a
    b += 42
    return b[0][0]

def augassign_add_2d_float(a : 'float[:,:]'):
    b = a
    b += 4.2
    return b[0][0]

def augassign_add_2d_complex(a : 'complex[:,:]'):
    b = a
    b += (4.0 + 2.0j)
    return b[0][0]

def augassign_add_sum_scalar(a : int, b : 'int[:]'):
    a += np.sum(b)
    return a

def augassign_add_sum_array(a : 'int[:,:]', b : 'int[:]'):
    a += np.sum(b)

def augassign_add_min_scalar(a : int, b : 'int[:]'):
    a += np.min(b)
    return a

def augassign_add_min_array(a : 'int[:,:]', b : 'int[:]'):
    a += np.min(b)

def augassign_add_norm_scalar(a : float, b : 'float[:]'):
    a += np.linalg.norm(b)
    return a

def augassign_add_norm_ord1_scalar(a : float, b : 'float[:]'):
    a += np.linalg.norm(b, ord=1)
    return a

# -=

def augassign_sub_1d_int(a : 'int[:]'):
    b = a
    b -= 42
    return b[0]

def augassign_sub_1d_float(a : 'float[:]'):
    b = a
    b -= 4.2
    return b[0]

def augassign_sub_1d_complex(a : 'complex[:]'):
    b = a
    b -= (4.0 + 2.0j)
    return b[0]

def augassign_sub_2d_int(a : 'int[:,:]'):
    b = a
    b -= 42
    return b[0][0]

def augassign_sub_2d_float(a : 'float[:,:]'):
    b = a
    b -= 4.2
    return b[0][0]

def augassign_sub_2d_complex(a : 'complex[:,:]'):
    b = a
    b -= (4.0 + 2.0j)
    return b[0][0]

def augassign_sub_sum_scalar(a : int, b : 'int[:]'):
    a -= np.sum(b)
    return a

def augassign_sub_max_scalar(a : int, b : 'int[:]'):
    a -= np.max(b)
    return a

def augassign_sub_max_array(a : 'int[:,:]', b : 'int[:]'):
    a -= np.max(b)

# *=

def augassign_mul_1d_int(a : 'int[:]'):
    b = a
    b *= 42
    return b[0]

def augassign_mul_1d_float(a : 'float[:]'):
    b = a
    b *= 4.2
    return b[0]

def augassign_mul_1d_complex(a : 'complex[:]'):
    b = a
    b *= (4.0 + 2.0j)
    return b[0]

def augassign_mul_2d_int(a : 'int[:,:]'):
    b = a
    b *= 42
    return b[0][0]

def augassign_mul_2d_float(a : 'float[:,:]'):
    b = a
    b *= 4.2
    return b[0][0]

def augassign_mul_2d_complex(a : 'complex[:,:]'):
    b = a
    b *= (4.0 + 2.0j)
    return b[0][0]

def augassign_mul_sum_scalar(a : int, b : 'int[:]'):
    a *= np.sum(b)
    return a

# /=

def augassign_div_1d_int(a : 'int[:]'):
    b = a
    b /= 42
    return b[0]

def augassign_div_1d_float(a : 'float[:]'):
    b = a
    b /= 4.2
    return b[0]

def augassign_div_1d_complex(a : 'complex[:]'):
    b = a
    b /= (4.0 + 2.0j)
    return b[0]

def augassign_div_2d_int(a : 'int[:,:]'):
    b = a
    b /= 42
    return b[0][0]

def augassign_div_2d_float(a : 'float[:,:]'):
    b = a
    b /= 4.2
    return b[0][0]

def augassign_div_2d_complex(a : 'complex[:,:]'):
    b = a
    b /= (4.0 + 2.0j)
    return b[0][0]

def augassign_func(x : float, y : float):
    def fun1(x: 'float') -> 'float':
        return x + 1
    x %= fun1(y)
    return x

def augassign_array_func(x : 'float[:]', y : 'float[:]'):
    def fun1(x: 'float[:]') -> 'float[:]':
        return x + 1
    x %= fun1(y)

def augassign_floor_div(a : 'float[:]'):
    a //= 3
