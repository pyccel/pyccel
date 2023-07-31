# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types

def create_complex_literal__int_int():
    a = complex(1,-2)
    return a

def create_complex_literal__int_float():
    a = complex(-1,2.7)
    return a

def create_complex_literal__int_complex():
    a = complex(1,-2j)
    b = complex(1,1+3j)
    return a, b

def create_complex_literal__float_int():
    a = complex(-1.6,2)
    return a

def create_complex_literal__float_float():
    a = complex(1.6,-2.9)
    return a

def create_complex_literal__float_complex():
    a = complex(1.6,2.8-7j)
    return a

def create_complex_literal__complex_int():
    a = complex(2.8-7j,1)
    return a

def create_complex_literal__complex_float():
    a = complex(-2.8+7j,1.8)
    return a

def create_complex_literal__complex_complex():
    a = complex(2.8+7j,-1.5-22j)
    return a

def cast_complex_literal():
    a = complex(2.8+7j)
    return a

def create_complex_var__int_int(a : 'int', b : 'int'):
    return complex(a,b)

def create_complex_var__int_complex(a : 'int', b : 'complex'):
    return complex(a,b)

def create_complex_var__complex_float(a : 'complex', b : 'float'):
    return complex(a,b)

def create_complex_var__complex_complex(a : 'complex', b : 'complex'):
    return complex(a,b)

def create_complex__int_int(a : 'int'):
    return complex(a,1), complex(1,a)

def create_complex_0__int_int(a : 'int'):
    return complex(a,0), complex(0,a)

def create_complex__float_float(a : 'float'):
    return complex(a,1.5), complex(1.5, a)

def create_complex_0__float_float(a : 'float'):
    return complex(a,0.0), complex(0.0, a)

def create_complex__complex_complex(a : 'complex'):
    return complex(a,1-2j), complex(1+2j,a)

def cast_complex_1(a : 'complex64'):
    return complex(a)

def cast_complex_2(a : 'complex128'):
    return complex(a)

def cast_float_complex(a : float, b : complex):
    return complex(a + b * 1j)

