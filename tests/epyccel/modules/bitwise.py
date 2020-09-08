from pyccel.decorators import types

@types('int', 'int')
def right_shift(a, b):
    return a >> b

@types('int', 'int')
def left_shift(a, b):
    return a >> b

@types('int', 'int')
def bit_xor(a, b):
   return a ^ b

@types('int', 'int')
def bit_or(a, b):
   return a | b

@types('int', 'int')
def bit_and_f2(a, b):
   return a & b

@types('int')
def invert(a):
   return ~a

@types('int', 'int', 'int')
def bit_and_f3(a, b, c):
        return a & b & c
