# pylint: disable=missing-function-docstring, missing-module-docstring

def right_shift_b_b(a : 'bool', b : 'bool'):
    return a >> b

def right_shift_i_i(a : 'int', b : 'int'):
    return a >> b

def right_shift_b_i(a : 'bool', b : 'int'):
    return a >> b

def left_shift_b_i(a : 'bool', b : 'int'):
    return a >> b

def left_shift_i_i(a : 'int', b : 'int'):
    return a >> b

def left_shift_b_b(a : 'bool', b : 'bool'):
    return a >> b

def bit_xor_b_b(a : 'bool', b : 'bool'):
   return a ^ b

def bit_xor_b_b_b(a : 'bool', b : 'bool', c : 'bool'):
    return a ^ b ^ c

def bit_xor_i_i(a : 'int', b : 'int'):
   return a ^ b

def bit_xor_b_i(a : 'bool', b : 'int'):
   return a ^ b

def bit_or_i_b(a : 'int', b : 'bool'):
   return a | b

def bit_or_i_i(a : 'int', b : 'int'):
   return a | b

def bit_or_b_b(a : 'bool', b : 'bool'):
   return a | b

def bit_and_i_b(a : 'int', b : 'bool'):
   return a & b

def bit_and_i_i(a : 'int', b : 'int'):
   return a & b

def bit_and_b_b(a : 'bool', b : 'bool'):
   return a & b

def bit_and_i_i_i(a : 'int', b : 'int', c : 'int'):
        return a & b & c

def bit_and_b_b_i(a : 'bool', b : 'bool', c : 'int'):
        return a & b & c

def invert_b(a : 'bool'):
   return ~a

def invert_i(a : 'int'):
   return ~a

def or_ints(n : int):
    if n & 1 or n < 128:
        return 1
    else:
        return 0
