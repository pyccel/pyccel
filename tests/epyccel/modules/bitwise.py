# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types('bool', 'bool')
def right_shift_b_b(a, b):
    return a >> b

@types('int', 'int')
def right_shift_i_i(a, b):
    return a >> b

@types('bool', 'int')
def right_shift_b_i(a, b):
    return a >> b

@types('bool', 'int')
def left_shift_b_i(a, b):
    return a >> b

@types('int', 'int')
def left_shift_i_i(a, b):
    return a >> b

@types('bool', 'bool')
def left_shift_b_b(a, b):
    return a >> b

@types('bool', 'bool')
def bit_xor_b_b(a, b):
   return a ^ b

@types('bool', 'bool', 'bool')
def bit_xor_b_b_b(a, b, c):
    return a ^ b ^ c

@types('int', 'int')
def bit_xor_i_i(a, b):
   return a ^ b

@types('bool', 'int')
def bit_xor_b_i(a, b):
   return a ^ b

@types('int', 'bool')
def bit_or_i_b(a, b):
   return a | b

@types('int', 'int')
def bit_or_i_i(a, b):
   return a | b

@types('bool', 'bool')
def bit_or_b_b(a, b):
   return a | b

@types('int', 'bool')
def bit_and_i_b(a, b):
   return a & b

@types('int', 'int')
def bit_and_i_i(a, b):
   return a & b

@types('bool', 'bool')
def bit_and_b_b(a, b):
   return a & b

@types('int', 'int', 'int')
def bit_and_i_i_i(a, b, c):
        return a & b & c

@types('bool', 'bool', 'int')
def bit_and_b_b_i(a, b, c):
        return a & b & c

@types('bool')
def invert_b(a):
   return ~a

@types('int')
def invert_i(a):
   return ~a
