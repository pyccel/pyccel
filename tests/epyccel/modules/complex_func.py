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

@types('int','int')
def create_complex_var__int_int(a,b):
    return complex(a,b)

@types('int','complex')
def create_complex_var__int_complex(a,b):
    return complex(a,b)

@types('complex', 'float')
def create_complex_var__complex_float(a,b):
    return complex(a,b)

@types('complex', 'complex')
def create_complex_var__complex_complex(a,b):
    return complex(a,b)

@types('int')
def create_complex__int_int(a):
    return complex(a,1), complex(1,a)

@types('int')
def create_complex_0__int_int(a):
    return complex(a,0), complex(0,a)

@types('float')
def create_complex__float_float(a):
    return complex(a,1.5), complex(1.5, a)

@types('float')
def create_complex_0__float_float(a):
    return complex(a,0.0), complex(0.0, a)

@types('complex')
def create_complex__complex_complex(a):
    return complex(a,1-2j), complex(1+2j,a)

@types('complex64')
def cast_complex_1(a):
    return complex(a)

@types('complex128')
def cast_complex_2(a):
    return complex(a)

def cast_float_complex(a : float, b : complex):
    return complex(a + b * 1j)

