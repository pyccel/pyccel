#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
This module contains all types which define a python class which is automatically recognised by pyccel
"""
from .builtins  import PythonImag, PythonReal, PythonConjugate
from .core      import ClassDef, PyccelFunctionDef
from .datatypes import (NativeBool, NativeInteger, NativeFloat,
                        NativeComplex, NativeString, NativeNumeric)
from .numpyext  import (NumpyShape, NumpySum, NumpyAmin, NumpyAmax,
                        NumpyImag, NumpyReal, NumpyTranspose,
                        NumpyConjugate, NumpySize, NumpyResultType)

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

ComplexClass = ClassDef('complex',
        methods=[
            PyccelFunctionDef('imag', func_class = PythonImag,
                decorators={'property':'property', 'numpy_wrapper':'numpy_wrapper'}),
            PyccelFunctionDef('real', func_class = PythonReal,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('conjugate', func_class = PythonConjugate,
                decorators={'numpy_wrapper': 'numpy_wrapper'}),
            ])

#=======================================================================================

FloatClass = ClassDef('float',
        methods=[
            PyccelFunctionDef('imag', func_class = PythonImag,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('real', func_class = PythonReal,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('conjugate', func_class = PythonConjugate,
                decorators={'numpy_wrapper': 'numpy_wrapper'}),
            #as_integer_ratio
            #fromhex
            #hex
            #is_integer
            ])

#=======================================================================================

IntegerClass = ClassDef('integer',
        methods=[
            PyccelFunctionDef('imag', func_class = PythonImag,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('real', func_class = PythonReal,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('conjugate', func_class = PythonConjugate,
                decorators={'numpy_wrapper': 'numpy_wrapper'}),
            #as_integer_ratio
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

NumpyArrayClass = ClassDef('numpy.ndarray',
        methods=[
            PyccelFunctionDef('shape', func_class = NumpyShape,
                decorators = {'property': 'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('size', func_class = NumpySize,
                decorators = {'property': 'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('T', func_class = NumpyTranspose,
                decorators = {'property': 'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('transpose', func_class = NumpyTranspose,
                decorators = {'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('sum', func_class = NumpySum,
                decorators = {'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('min', func_class = NumpyAmin,
                decorators = {'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('max', func_class = NumpyAmax,
                decorators = {'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('imag', func_class = NumpyImag,
                decorators = {'property': 'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('real', func_class = NumpyReal,
                decorators = {'property': 'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('conj', func_class = NumpyConjugate,
                decorators = {'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('conjugate', func_class = NumpyConjugate,
                decorators = {'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('dtype', func_class = NumpyResultType,
                decorators = {'property': 'property', 'numpy_wrapper': 'numpy_wrapper'}),
        ]
)

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

