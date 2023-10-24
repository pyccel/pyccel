#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
This module contains all types which define a python class which is automatically recognised by pyccel
"""
from .core      import ClassDef, PyccelFunctionDef
from .datatypes import (NativeBool, NativeInteger, NativeFloat,
                        NativeComplex, NativeString, NativeNumeric)

__all__ = ('BooleanClass',
        'IntegerClass',
        'RealClass',
        'ComplexClass',
        'StringClass',
        'NumpyArrayClass',
        'TupleClass',
        'literal_classes',
        'get_cls_base')

#=======================================================================================

ComplexClass = ClassDef('complex', methods=[])

#=======================================================================================

FloatClass = ClassDef('float',
        methods=[
            #as_integer_ratio
            #fromhex
            #hex
            #is_integer
            ])

#=======================================================================================

IntegerClass = ClassDef('integer',
        methods=[
            #as_integer_ratio
            #bit_count
            #bit_length
            #denominator
            #from_bytes
            #numerator
            #to_bytes
            ])

#=======================================================================================

BooleanClass = ClassDef('boolean',
        superclasses=(IntegerClass,))

#=======================================================================================

StringClass = ClassDef('string',
        methods=[
                #capitalize
                #casefold
                #center
                #count
                #encode
                #endswith
                #expandtabs
                #find
                #format
                #format_map
                #index
                #isalnum
                #isalpha
                #isascii
                #isdecimal
                #isdigit
                #isidentifier
                #islower
                #isnumeric
                #isprintable
                #isspace
                #istitle
                #isupper
                #join
                #ljust
                #lower
                #lstrip
                #maketrans
                #partition
                #replace
                #rfind
                #rindex
                #rjust
                #rpartition
                #rsplit
                #rstrip
                #split
                #splitlines
                #startswith
                #strip
                #swapcase
                #title
                #translate
                #upper
                #zfill
                ])

#=======================================================================================

TupleClass = ClassDef('tuple',
        methods=[
            #index
            #count
            ])

#=======================================================================================

NumpyArrayClass = ClassDef('numpy.ndarray', methods=[])

#=======================================================================================

literal_classes = {
        NativeBool()    : BooleanClass,
        NativeInteger() : IntegerClass,
        NativeFloat()   : FloatClass,
        NativeComplex() : ComplexClass,
        NativeString()  : StringClass
}

#=======================================================================================

def get_cls_base(dtype, precision, rank):
    """
    Determine the base class of an object.

    From the dtype and rank, determine the base class of an object.

    Parameters
    ----------
    dtype : DataType
        The data type of the object.

    precision : int
        The precision of the object.

    rank : int
        The rank of the object.

    Returns
    -------
    ClassDef
        A class definition describing the base class of an object.

    Raises
    ------
    NotImplementedError
        Raised if the base class cannot be found.
    """
    if precision in (-1, 0, None) and rank == 0:
        return literal_classes[dtype]
    elif dtype in NativeNumeric:
        return NumpyArrayClass
    else:
        type_name = f"{dtype}({precision})"
        if rank:
            dims = ','.join(':' for _ in range(rank))
            type_name += f"[{dims}]"
        raise NotImplementedError(f"No class definition found for type {type_name}")

