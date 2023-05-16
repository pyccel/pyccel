#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
This module contains all types which define a python class which is automatically recognised by pyccel
"""
from .builtins  import PythonImag, PythonReal, PythonConjugate
from .core      import ClassDef, FunctionDef
from .datatypes import (NativeBool, NativeInteger, NativeFloat,
                        NativeComplex, NativeString)
from .numpyext  import (NumpyShape, NumpySum, NumpyAmin, NumpyAmax,
                        NumpyImag, NumpyReal, NumpyTranspose,
                        NumpyConjugate, NumpySize)

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
            FunctionDef('imag',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':PythonImag}),
            FunctionDef('real',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':PythonReal}),
            FunctionDef('conjugate',[],[],body=[],
                decorators={'numpy_wrapper':PythonConjugate}),
            ])

#=======================================================================================

FloatClass = ClassDef('float',
        methods=[
            FunctionDef('imag',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':PythonImag}),
            FunctionDef('real',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':PythonReal}),
            #conjugate
            #as_integer_ratio
            #fromhex
            #hex
            #is_integer
            ])

#=======================================================================================

IntegerClass = ClassDef('integer',
        methods=[
            FunctionDef('imag',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':PythonImag}),
            FunctionDef('real',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':PythonReal}),
            #as_integer_ratio
            #bit_length
            #conjugate
            #denominator
            #from_bytes
            #numerator
            #to_bytes
            ])

#=======================================================================================

BooleanClass = ClassDef('boolean',
        superclass=(IntegerClass,))

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
            FunctionDef('shape',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':NumpyShape}),
            FunctionDef('size',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':NumpySize}),
            FunctionDef('T',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':NumpyTranspose}),
            FunctionDef('transpose',[],[],body=[],
                decorators={'numpy_wrapper':NumpyTranspose}),
            FunctionDef('sum',[],[],body=[],
                decorators={'numpy_wrapper':NumpySum}),
            FunctionDef('min',[],[],body=[],
                decorators={'numpy_wrapper':NumpyAmin}),
            FunctionDef('max',[],[],body=[],
                decorators={'numpy_wrapper':NumpyAmax}),
            FunctionDef('imag',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':NumpyImag}),
            FunctionDef('real',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':NumpyReal}),
            FunctionDef('conj',[],[],body=[],
                decorators={'numpy_wrapper':NumpyConjugate}),
            FunctionDef('conjugate',[],[],body=[],
                decorators={'numpy_wrapper':NumpyConjugate})])

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
    From the dtype and rank, determine the base class of an object
    """
    if precision == -1 and rank == 0:
        return literal_classes[dtype]
    else:
        return NumpyArrayClass

