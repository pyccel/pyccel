#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
This module contains all types which define a python class which is automatically recognised by pyccel
"""

from pyccel.ast.builtin_methods.set_methods  import (SetAdd, SetClear, SetCopy, SetPop,
                                                     SetDiscard, SetUpdate, SetUnion,
                                                     SetIntersection, SetIntersectionUpdate,
                                                     SetIsDisjoint, SetDifference,
                                                     SetDifferenceUpdate)
from pyccel.ast.builtin_methods.list_methods import (ListAppend, ListInsert, ListPop,
                                                     ListClear, ListExtend, ListRemove,
                                                     ListCopy, ListSort, ListReverse)
from pyccel.ast.builtin_methods.dict_methods import (DictPop, DictPopitem, DictGet, DictClear,DictCopy,
                                                     DictSetDefault, DictItems, DictGetItem, DictKeys,
                                                     DictValues)

from .builtins   import PythonImag, PythonReal, PythonConjugate
from .core       import ClassDef, PyccelFunctionDef
from .datatypes  import (PythonNativeBool, PythonNativeInt, PythonNativeFloat,
                         PythonNativeComplex, StringType, TupleType, CustomDataType,
                         HomogeneousListType, HomogeneousSetType, DictType, SymbolicType,
                         GenericType)
from .numpyext   import (NumpyShape, NumpySum, NumpyAmin, NumpyAmax,
                         NumpyImag, NumpyReal, NumpyTranspose,
                         NumpyConjugate, NumpySize, NumpyResultType, NumpyArray)
from .numpytypes import NumpyNumericType, NumpyNDArrayType

__all__ = (
    'BooleanClass',
    'ComplexClass',
    'DictClass',
    'FloatClass',
    'IntegerClass',
    'ListClass',
    'NumpyArrayClass',
    'SetClass',
    'StringClass',
    'TupleClass',
    'get_cls_base',
    'literal_classes',
)

#=======================================================================================

ComplexClass = ClassDef('complex', class_type = PythonNativeComplex(),
        methods=[
            PyccelFunctionDef('imag', func_class = PythonImag,
                decorators={'property':'property', 'numpy_wrapper':'numpy_wrapper'}),
            PyccelFunctionDef('real', func_class = PythonReal,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}),
            PyccelFunctionDef('conjugate', func_class = PythonConjugate,
                decorators={'numpy_wrapper': 'numpy_wrapper'}),
            ])

#=======================================================================================

FloatClass = ClassDef('float', class_type = PythonNativeFloat(),
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

IntegerClass = ClassDef('integer', class_type = PythonNativeInt(),
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

BooleanClass = ClassDef('boolean', class_type = PythonNativeBool(),
        superclasses=(IntegerClass,))

#=======================================================================================

StringClass = ClassDef('string', class_type = StringType(),
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

ListClass = ClassDef('list',
        methods=[
            PyccelFunctionDef('append', func_class = ListAppend),
            PyccelFunctionDef('clear', func_class = ListClear),
            PyccelFunctionDef('copy', func_class = ListCopy),
            PyccelFunctionDef('extend', func_class = ListExtend),
            PyccelFunctionDef('insert', func_class = ListInsert),
            PyccelFunctionDef('pop', func_class = ListPop),
            PyccelFunctionDef('sort', func_class = ListSort),
            PyccelFunctionDef('remove', func_class = ListRemove),
            PyccelFunctionDef('reverse', func_class = ListReverse),
        ])

#=======================================================================================

SetClass = ClassDef('set',
        methods=[
            PyccelFunctionDef('add', func_class = SetAdd ),
            PyccelFunctionDef('clear', func_class = SetClear),
            PyccelFunctionDef('copy', func_class = SetCopy),
            PyccelFunctionDef('difference', func_class = SetDifference),
            PyccelFunctionDef('difference_update', func_class = SetDifferenceUpdate),
            PyccelFunctionDef('discard', func_class = SetDiscard),
            PyccelFunctionDef('intersection', func_class = SetIntersection),
            PyccelFunctionDef('intersection_update', func_class = SetIntersectionUpdate),
            PyccelFunctionDef('isdisjoint', func_class = SetIsDisjoint),
            PyccelFunctionDef('pop', func_class = SetPop),
            PyccelFunctionDef('remove', func_class = SetDiscard),
            PyccelFunctionDef('union', func_class = SetUnion),
            PyccelFunctionDef('update', func_class = SetUpdate),
            PyccelFunctionDef('__or__', func_class = SetUnion),
            PyccelFunctionDef('__and__', func_class = SetIntersection),
            PyccelFunctionDef('__iand__', func_class = SetIntersectionUpdate),
            PyccelFunctionDef('__sub__', func_class = SetDifference),
            PyccelFunctionDef('__isub__', func_class = SetDifferenceUpdate),
            PyccelFunctionDef('__ior__', func_class = SetUpdate),
        ])

#=======================================================================================

DictClass = ClassDef('dict',
        methods=[
            PyccelFunctionDef('copy', func_class = DictCopy),
            PyccelFunctionDef('clear', func_class = DictClear),
            PyccelFunctionDef('get', func_class = DictGet),
            PyccelFunctionDef('items', func_class = DictItems),
            PyccelFunctionDef('keys', func_class = DictKeys),
            PyccelFunctionDef('pop', func_class = DictPop),
            PyccelFunctionDef('popitem', func_class = DictPopitem),
            PyccelFunctionDef('setdefault', func_class = DictSetDefault),
            PyccelFunctionDef('values', func_class = DictValues),
            PyccelFunctionDef('__getitem__', func_class = DictGetItem),
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
            PyccelFunctionDef('copy', func_class = NumpyArray, argument_description = {'self': None, 'order':'C'},
                decorators = {'numpy_wrapper': 'numpy_wrapper'}),
        ]
)

#=======================================================================================

StackArrayClass = ClassDef('stack_array')

#=======================================================================================

literal_classes = {
        PythonNativeBool()    : BooleanClass,
        PythonNativeInt()     : IntegerClass,
        PythonNativeFloat()   : FloatClass,
        PythonNativeComplex() : ComplexClass,
        StringType()          : StringClass
}

#=======================================================================================

def get_builtin_cls_base(class_type):
    """
    Determine the base class of an object with a built-in datatype.

    From the type, determine the base class of an object with a built-in datatype.

    Parameters
    ----------
    class_type : DataType
        The Python type of the object.

    Returns
    -------
    ClassDef
        A class definition describing the base class of an object.

    Raises
    ------
    NotImplementedError
        Raised if the base class cannot be found.
    """
    if isinstance(class_type, (SymbolicType, GenericType)):
        return None
    elif class_type in literal_classes:
        return literal_classes[class_type]
    elif isinstance(class_type, (NumpyNumericType, NumpyNDArrayType)):
        return NumpyArrayClass
    elif isinstance(class_type, TupleType):
        return TupleClass
    elif isinstance(class_type, HomogeneousListType):
        return ListClass
    elif isinstance(class_type, HomogeneousSetType):
        return SetClass
    elif isinstance(class_type, DictType):
        return DictClass
    else:
        raise NotImplementedError(f"No class definition found for type {class_type}")

