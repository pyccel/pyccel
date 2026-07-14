# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

import numpy as np

IT = TypeVar("IT", bool, np.int8, np.int64)
IT2 = TypeVar("IT2", bool, np.int8, np.int64)

def numpy_bit_and_1(a: "IT[:,:,:]", b: "IT2[:,:,:]"):
    return a & b

def numpy_bit_and_2(a: "IT[:,:,:]", b: "IT2"):
    return a & b

def numpy_bit_and_3(a: "IT", b: "IT2[:,:,:]"):
    return a & b

def numpy_bit_or_1(a: "IT[:,:,:]", b: "IT2[:,:,:]"):
    return a | b

def numpy_bit_or_2(a: "IT[:,:,:]", b: "IT2"):
    return a | b

def numpy_bit_or_3(a: "IT", b: "IT2[:,:,:]"):
    return a | b

def numpy_bit_xor_1(a: "IT[:,:,:]", b: "IT2[:,:,:]"):
    return a ^ b

def numpy_bit_xor_2(a: "IT[:,:,:]", b: "IT2"):
    return a ^ b

def numpy_bit_xor_3(a: "IT", b: "IT2[:,:,:]"):
    return a ^ b

def numpy_bit_lshift_1(a: "IT[:,:,:]", b: "IT2[:,:,:]"):
    return a << b

def numpy_bit_lshift_2(a: "IT[:,:,:]", b: "IT2"):
    return a << b

def numpy_bit_lshift_3(a: "IT", b: "IT2[:,:,:]"):
    return a << b

def numpy_bit_rshift_1(a: "IT[:,:,:]", b: "IT2[:,:,:]"):
    return a >> b

def numpy_bit_rshift_2(a: "IT[:,:,:]", b: "IT2"):
    return a >> b

def numpy_bit_rshift_3(a: "IT", b: "IT2[:,:,:]"):
    return a >> b

def numpy_bit_invert(a: "IT[:,:,:]"):
    return ~a
