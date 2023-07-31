# pylint: disable=missing-function-docstring, missing-module-docstring

from pyccel.decorators import types

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
