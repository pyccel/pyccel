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
                        NativeComplex, NativeString, NativeNumeric,NativeList,
                        NativeTuple, CustomDataType)
from .numpyext  import (NumpyShape, NumpySum, NumpyAmin, NumpyAmax,
                        NumpyImag, NumpyReal, NumpyTranspose,
                        NumpyConjugate, NumpySize, NumpyResultType,
                        NumpyArray, NumpyNDArrayType)

__all__ = ('BooleanClass',
        'IntegerClass',
        'FloatClass',
        'ComplexClass',
        'StringClass',
        'NumpyArrayClass',
        'ListClassDef',
        'TupleClass',
        'literal_classes',
        'get_cls_base')
#=======================================================================================

class ListClassDef(ClassDef):
    def __init__(self, name, elements_type=None):
        super().__init__(name, class_type=elements_type if elements_type else NativeList,
                         methods=[
                             PyccelFunctionDef('__init__', argument_description={'self': None, 'values': None}),
                             PyccelFunctionDef('append', argument_description={'self': None, 'value': None}),
                             PyccelFunctionDef('pop', argument_description={'self': None, 'index': None}),
                             PyccelFunctionDef('insert', argument_description={'self': None, 'index': None, 'value': None}),
                             PyccelFunctionDef('__str__', argument_description={'self': None}),
                             PyccelFunctionDef('slice_method', argument_description={'self': None, 'start': None, 'stop': None, 'step': None}),
                             PyccelFunctionDef('clear', argument_description={'self': None}),
                             PyccelFunctionDef('extend', argument_description={'self': None, 'iterable': None}),
                             PyccelFunctionDef('reverse', argument_description={'self': None}),
                             PyccelFunctionDef('sort', argument_description={'self': None,'key':None,'reverse':None}),
                             PyccelFunctionDef('remove', argument_description={'self': None,'value':None}),
                             PyccelFunctionDef('index', argument_description={'self': None,'value':None}),
                             PyccelFunctionDef('count', argument_description={'self': None,'value':None}),
                         ])

    def __str__(self):
        return f"class {self.name}({self.class_type}):"

# Example of adding list methods
list_class = ListClassDef(name='list', elements_type=int)
list_class.methods.extend([
    PyccelFunctionDef('__init__', argument_description={'self': None, 'values': None}),
    PyccelFunctionDef('append', argument_description={'self': None, 'value': None}),
    PyccelFunctionDef('pop', argument_description={'self': None, 'index': None}),
    PyccelFunctionDef('insert', argument_description={'self': None, 'index': None, 'value': None}),
    PyccelFunctionDef('__str__', argument_description={'self': None}),
    PyccelFunctionDef('slice_method', argument_description={'self': None, 'start': None, 'stop': None, 'step': None}),
    PyccelFunctionDef('clear', argument_description={'self': None}),
    PyccelFunctionDef('extend', argument_description={'self': None, 'iterable': None}),
    PyccelFunctionDef('reverse', argument_description={'self': None}),
    PyccelFunctionDef('sort', argument_description={'self': None,'key':None,'reverse':None}),
    PyccelFunctionDef('remove', argument_description={'self': None,'value':None}),
    PyccelFunctionDef('index', argument_description={'self': None,'value':None}),
    PyccelFunctionDef('count', argument_description={'self': None,'value':None}),
])

#=======================================================================================

ComplexClass = ClassDef('complex', class_type = NativeComplex(),
        methods=[
            PyccelFunctionDef('imag', func_class = PythonImag,
                decorators={'property':'property', 'numpy_wrapper':'numpy_wrapper'}),
            PyccelFunctionDef('real', func_class = PythonReal,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('conjugate', func_class = PythonConjugate,
                decorators={'numpy_wrapper': 'numpy_wrapper'}),
            ])

#=======================================================================================

FloatClass = ClassDef('float', class_type = NativeFloat(),
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

IntegerClass = ClassDef('integer', class_type = NativeInteger(),
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

BooleanClass = ClassDef('boolean', class_type = NativeBool(),
        superclasses=(IntegerClass,))

#=======================================================================================

StringClass = ClassDef('string', class_type = NativeString(),
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

TupleClass = ClassDef('tuple', class_type = NativeTuple(),
        methods=[
            #index
            #count
            ])

#=======================================================================================

NumpyArrayClass = ClassDef('numpy.ndarray', class_type = NumpyNDArrayType(),
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
            PyccelFunctionDef('copy', func_class = NumpyArray, argument_description = {'self': None, 'order':'C'},
                decorators = {'numpy_wrapper': 'numpy_wrapper'}),
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

def get_cls_base(dtype, precision, container_type):
    """
    Determine the base class of an object.

    From the dtype and rank, determine the base class of an object.

    Parameters
    ----------
    dtype : DataType
        The data type of the object.

    precision : int
        The precision of the object.

    container_type : DataType
        The Python type of the object. If this is different to the dtype then
        the object is a container.

    Returns
    -------
    ClassDef
        A class definition describing the base class of an object.

    Raises
    ------
    NotImplementedError
        Raised if the base class cannot be found.
    """
    if isinstance(dtype, CustomDataType) and container_type is dtype:
        return None
    if precision in (-1, 0, None) and container_type is dtype:
        return literal_classes[dtype]
    elif dtype in NativeNumeric or isinstance(container_type, NumpyNDArrayType):
        return NumpyArrayClass
    elif isinstance(container_type, NativeTuple):
        return TupleClass
    else:
        if container_type:
            type_name = str(container_type)
        else:
            type_name = f"{dtype}({precision})"
        raise NotImplementedError(f"No class definition found for type {type_name}")

